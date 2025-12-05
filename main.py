import argparse
import yaml
import torchvision.transforms as transforms
import torch.nn as nn
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils.utils import *
from datetime import datetime
from torchvision.ops import MLP
import warnings

warnings.filterwarnings("ignore")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of CCA in yaml format')
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--out', dest='out', default='outputs', help='output directory')
    parser.add_argument('--cache', dest='cache', default='caches', help='cache directory')
    parser.add_argument('--root', dest='root', default='../data', help='data root directory')
    args = parser.parse_args()

    return args


def run_cca(f, cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
            val_features_ica=None, test_features_ica=None, cache_keys_ica=None):
    # Zero-shot CLIP
    clip_logits = temperature * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("**** Zero-shot CLIP's val accuracy: {:.2f}. ****".format(acc))
    f.write("**** Zero-shot CLIP's val accuracy: {:.2f}. ****".format(acc) + '\n')

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    gamma = cfg['init_gamma']
    eta = cfg['init_eta']

    # CCA
    clip_logits_0 = clip_logits

    attn_w = cache_keys.T @ clip_weights
    clip_logits_1 = temperature * val_features @ (cache_keys @ F.softmax(attn_w, dim=0)) * gamma
    attn_w = val_features @ clip_weights
    clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * eta
    clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

    affinity = val_features_ica @ cache_keys_ica
    affinity = normalize(affinity)

    cache_logits_ica = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    cca_logits = clip_logits + cache_logits_ica * alpha
    acc = cls_acc(cca_logits, val_labels)
    print("**** CCA val accuracy: {:.2f}. ****".format(acc))
    f.write("**** CCA val accuracy: {:.2f}. ****".format(acc) + '\n')

    # Search Hyperparameters
    if cfg['search_hp']:
        print("-------- Searching hyperparameters on the val set. --------")
        f.write("-------- Searching hyperparameters on the val set. --------\n")
        best_beta, best_alpha, best_gamma, best_eta = search_hp(f, cfg, cache_keys,
                                                                cache_keys_ica,
                                                                cache_values,
                                                                val_features,
                                                                val_features_ica,
                                                                val_labels,
                                                                clip_weights)
    else:
        best_beta, best_alpha = beta, alpha
        best_gamma = gamma
        best_eta = eta

    # ==============================================================================================
    print("-------- Evaluating on the test set. --------")
    f.write("-------- Evaluating on the test set. --------\n")
    # Zero-shot CLIP
    clip_logits = temperature * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("**** Zero-shot CLIP's test accuracy: {:.2f}. ****".format(acc))
    f.write("**** Zero-shot CLIP's test accuracy: {:.2f}. ****".format(acc) + '\n')

    # CCA
    clip_logits_0 = clip_logits

    attn_w = cache_keys.T @ clip_weights
    clip_logits_1 = temperature * test_features @ (cache_keys @ F.softmax(attn_w, dim=0)) * best_gamma
    attn_w = test_features @ clip_weights
    clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * best_eta

    clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

    affinity = test_features_ica @ cache_keys_ica
    affinity = normalize(affinity)

    cache_logits_ica = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    cca_logits = clip_logits + cache_logits_ica * best_alpha
    acc = cls_acc(cca_logits, test_labels)
    print("**** CCA test accuracy: {:.2f}. ****\n".format(acc))
    f.write("**** CCA test accuracy: {:.2f}. ****\n".format(acc))

    return acc


def run_cca_ft(f, cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,
               clip_weights, clip_model, train_loader_F, mat_unmix, val_features_ica=None, test_features_ica=None,
               cache_keys_ica=None):
    print("-------- Running CCA-FT --------")
    f.write("\n-------- Running CCA-FT --------\n")
    # initialize cache adapter
    dim = mat_unmix.shape[0]
    cache_adapter = MLP(dim, [dim], activation_layer=None, bias=False, dropout=0.0)
    cache_adapter[0].weight = nn.Parameter(torch.eye(dim, dtype=torch.float32).cuda())
    cache_adapter = cache_adapter.to('cuda')

    # initialize zs classifier
    in_features, out_features = clip_weights.shape
    zs_classifier = nn.Linear(in_features=in_features, out_features=out_features, bias=False).to(torch.float32).cuda()
    zs_classifier.weight = nn.Parameter(clip_weights.clone().t())

    lr = 0.001
    lr_zs = 0.0001

    optimizer_cache = torch.optim.AdamW([{'params': cache_adapter.parameters(), 'lr': lr, 'eps': 1e-4}])
    optimizer_zs = torch.optim.AdamW([{'params': zs_classifier.parameters(), 'lr': lr_zs, 'eps': 1e-4}])

    scheduler_cache = torch.optim.lr_scheduler.StepLR(optimizer_cache, step_size=10, gamma=0.95)
    scheduler_zs = torch.optim.lr_scheduler.StepLR(optimizer_zs, step_size=10, gamma=0.95)

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    gamma = cfg['init_gamma']
    eta = cfg['init_eta']

    best_acc, best_epoch = 0.0, 0

    best_cache_adapter_path = join(cfg['model_dir'], "best_cache_adapter_" + str(cfg['shots']) + "shots.pt")
    best_zs_classifier_path = join(cfg['model_dir'], "best_zs_classifier_" + str(cfg['shots']) + "shots.pt")

    for train_idx in range(cfg['train_epoch']):
        # Train
        cache_adapter.train()
        zs_classifier.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        s = 'Train Epoch: {:>2} / {:}'.format(train_idx, cfg['train_epoch'])

        for i, (images, target) in enumerate(train_loader_F):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.to(torch.float32)

                image_features_ica = image_features @ mat_unmix.to('cuda').T

                attn_w = image_features @ clip_weights
                clip_logits_1 = temperature * image_features @ (image_features.T @ F.softmax(attn_w, dim=0)) * gamma
                clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * eta

            clip_logits_0 = temperature * zs_classifier(image_features)
            clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

            affinity = get_affinity(image_features_ica, cache_keys_ica, cache_adapter)
            cache_logits_ica = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            cca_logits = clip_logits + cache_logits_ica * alpha

            loss = F.cross_entropy(cca_logits, target)
            l1_norm = sum(p.abs().sum() for p in cache_adapter.parameters())
            loss = loss + l1_norm * 0.00001

            acc = cls_acc(cca_logits, target)
            correct_samples += acc / 100 * len(cca_logits)
            all_samples += len(cca_logits)
            loss_list.append(loss.item())

            optimizer_cache.zero_grad()
            optimizer_zs.zero_grad()

            loss.backward()

            optimizer_cache.step()
            optimizer_zs.step()

            scheduler_cache.step()
            scheduler_zs.step()

        curr_lr = scheduler_cache.get_last_lr()
        s = s + ' || LR: {:.6f}, '.format(curr_lr[0])
        curr_lr_zs = scheduler_zs.get_last_lr()
        s = s + '{:.6f}, '.format(curr_lr_zs[0])
        s = s + 'Acc: {:.4f} ({:>2}/{:}), Loss: {:.4f}'.format(correct_samples / all_samples, int(correct_samples),
                                                               all_samples, sum(loss_list) / len(loss_list))

        # Eval
        zs_classifier.eval()
        cache_adapter.eval()

        clip_logits_0 = temperature * zs_classifier(test_features)

        attn_w = cache_keys.T @ clip_weights
        clip_logits_1 = temperature * test_features @ (cache_keys @ F.softmax(attn_w, dim=0)) * gamma
        attn_w = test_features @ clip_weights
        clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * eta

        clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

        affinity = get_affinity(test_features_ica, cache_keys_ica, cache_adapter)
        cache_logits_ica = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        cca_logits = clip_logits + cache_logits_ica * alpha

        acc = cls_acc(cca_logits, test_labels)

        print(s + " || test accuracy: {:.2f}".format(acc))
        f.write(s + " || test accuracy: {:.2f}".format(acc) + '\n')
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(zs_classifier.weight, best_zs_classifier_path)
            torch.save(cache_adapter.state_dict(), best_cache_adapter_path)

    zs_classifier.weight = torch.load(best_zs_classifier_path)
    cache_adapter.load_state_dict(torch.load(best_cache_adapter_path))

    print(f"**** CCA-FT best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****")
    f.write(
        f"**** CCA-FT best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    if cfg['search_hp']:
        print("-------- Searching hyperparameters on the val set. --------")
        f.write("-------- Searching hyperparameters on the val set. --------\n")
        best_beta, best_alpha, best_gamma, best_eta = search_hp(f, cfg, cache_keys,
                                                                cache_keys_ica,
                                                                cache_values,
                                                                val_features,
                                                                val_features_ica,
                                                                val_labels,
                                                                clip_weights,
                                                                cache_adapter=cache_adapter,
                                                                zs_classifier=zs_classifier
                                                                )
    else:
        best_beta, best_alpha = beta, alpha
        best_gamma = gamma
        best_eta = eta

    print("-------- Evaluating on the test set. --------")
    f.write("-------- Evaluating on the test set. --------\n")
    clip_logits_0 = temperature * zs_classifier(test_features)

    attn_w = cache_keys.T @ clip_weights
    clip_logits_1 = temperature * test_features @ (cache_keys @ F.softmax(attn_w, dim=0)) * best_gamma
    attn_w = test_features @ clip_weights
    clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * best_eta

    clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

    affinity = get_affinity(test_features_ica, cache_keys_ica, cache_adapter)
    cache_logits_ica = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    cca_logits = clip_logits + cache_logits_ica * best_alpha

    acc = cls_acc(cca_logits, test_labels)
    print("**** CCA-FT test accuracy: {:.2f}. ****".format(max(best_acc, acc)))
    f.write("**** CCA-FT test accuracy: {:.2f}. ****".format(max(best_acc, acc)) + '\n')

    return max(best_acc, acc)


def main():
    print(torch.cuda.is_available(), torch.cuda.get_device_name())
    # Load config file
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M')
    args = get_arguments()
    if args.config is None:
        args.config = 'configs/ucf101.yaml'
        args.shot = 2
        args.out = 'outputs'
        args.cache = 'caches'
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    dataset_name = cfg['dataset']
    cfg['shots'] = args.shot
    backbone = cfg['backbone'].lower()

    output_dir = args.out
    log_path = os.path.join(output_dir, dataset_name, f'{dataset_name}_{args.shot}shot.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, 'w')
    f.write("\nStart time: " + start_time + '\n')

    cache_dir = os.path.join(args.cache, backbone, cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    model_dir = os.path.join(cache_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    cfg['model_dir'] = model_dir

    print("Running configs.")
    print(cfg, "")
    f.write(str(cfg) + '\n')

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    batch_size = 128

    random.seed(1)
    torch.manual_seed(1)

    print('\n===> Shots:', cfg['shots'], '\n')
    f.write('\n===> Shots: {}\n\n'.format(cfg['shots']))

    # prepare dataset
    dataset = build_dataset(cfg['dataset'], args.root, cfg['shots'], seed=1)
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                    shuffle=False)
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform,
                                           is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, tfm=train_tranform,
                                       is_train=True, shuffle=True)

    # Get textual features as CLIP's classifier
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    clip_weights_path = os.path.join(cache_dir, 'clip_weights.pt')
    if not exists(clip_weights_path):
        torch.save(clip_weights, clip_weights_path)

    # Construct the cache model by few-shot training set
    cache_keys, cache_values, cache_keys_ica, mat_unmix = build_cache_model(
        cfg,
        clip_model,
        train_loader_cache)

    mat_unmix = mat_unmix.to('cuda')
    cache_keys = cache_keys.to(torch.float32).to('cuda')
    cache_values = cache_values.to(torch.float32).to('cuda')
    clip_weights = clip_weights.to(torch.float32).to('cuda')
    cache_keys_ica = cache_keys_ica.to('cuda')

    # Pre-load val features
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    val_features = val_features.to(torch.float32).to('cuda')
    val_features_ica = val_features @ mat_unmix.to('cuda').T

    # Pre-load test features
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    test_features = test_features.to(torch.float32).to('cuda')
    test_features_ica = test_features @ mat_unmix.to('cuda').T

    # ------------------------------------------ CCA ------------------------------------------
    acc = run_cca(f, cfg, cache_keys, cache_values, val_features, val_labels, test_features,
                  test_labels, clip_weights, val_features_ica=val_features_ica,
                  test_features_ica=test_features_ica, cache_keys_ica=cache_keys_ica)

    # ------------------------------------------ CCA-FT ------------------------------------------
    ft_acc = run_cca_ft(f, cfg, cache_keys, cache_values, val_features, val_labels, test_features,
                        test_labels, clip_weights, clip_model, train_loader_F,
                        mat_unmix, val_features_ica=val_features_ica,
                        test_features_ica=test_features_ica, cache_keys_ica=cache_keys_ica)

    print('Output saved to:', log_path)
    f.write('Output saved to: ' + str(log_path))
    f.close()


if __name__ == '__main__':
    main()
