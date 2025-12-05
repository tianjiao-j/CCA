import torch
import torch.nn.functional as F
import clip
import os
from tqdm import tqdm
import itertools
from os.path import dirname, exists, join
import numpy as np
import random

torch.manual_seed(1)
random.seed(1)
temperature = 100.


def get_affinity(features, cache_keys, adapter):
    if adapter:
        affinity = features @ adapter(cache_keys.T).T
    else:
        affinity = features @ cache_keys
    affinity = normalize(affinity)

    return affinity


def normalize(affinity):
    if affinity.dim() == 1:
        x_min = affinity.min()
        x_max = affinity.max()
    else:
        x_min = affinity.min(dim=1, keepdim=True)[0]
        x_max = affinity.max(dim=1, keepdim=True)[0]
    affinity = (affinity - x_min) / (x_max - x_min)

    return affinity


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    key_path = join(cfg['cache_dir'], 'keys_' + str(cfg['shots']) + "shots.pt")
    value_path = join(cfg['cache_dir'], 'values_' + str(cfg['shots']) + "shots.pt")

    n_components = 1024
    mat_unmix_path = join(dirname(cfg['cache_dir']), f'mat_unmix_{str(n_components)}d.pt')

    load_cache = cfg['load_cache']
    has_key_value = exists(key_path) and exists(value_path)
    build_cache = not load_cache or not has_key_value

    if build_cache:
        print('Creating cache key and values...')
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                for i, (images, target) in enumerate(train_loader_cache):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)

            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
    else:
        print('Loading cache key and values...')
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)

    print('Loading unmixing matrix from {}'.format(mat_unmix_path))
    mat_unmix = torch.load(mat_unmix_path).to(torch.float32).cuda()
    cache_keys_ica = (cache_keys.T.to(torch.float32).to('cuda') @ mat_unmix.T).T

    if not exists(value_path):
        print('Saving cache values to {}'.format(value_path))
        torch.save(cache_values, value_path)
    if not exists(key_path):
        print('Saving cache keys to {}'.format(key_path))
        torch.save(cache_keys, key_path)

    return cache_keys, cache_values, cache_keys_ica, mat_unmix


def pre_load_features(cfg, split, clip_model, loader):
    feature_path = join(cfg['cache_dir'], split + "_f.pt")
    label_path = join(cfg['cache_dir'], split + "_l.pt")

    if cfg['load_pre_feat'] == False or not exists(feature_path) or not exists(label_path):
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, feature_path)
        torch.save(labels, label_path)

    else:
        features = torch.load(feature_path)
        labels = torch.load(label_path)

    return features, labels


def search_hp(f, cfg, cache_keys, cache_keys_ica, cache_values, features, features_ica, labels,
              clip_weights, cache_adapter=None, zs_classifier=None):
    if cfg['search_hp']:
        with torch.no_grad():
            best_beta, best_alpha = cfg['init_beta'], cfg['init_alpha']
            best_gamma = cfg['init_gamma']
            best_eta = cfg['init_eta']

            beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                         range(cfg['search_step'][0])]
            alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                          range(cfg['search_step'][1])]
            gamma_list = np.linspace(best_gamma / 5., best_gamma * 5., num=100).tolist()
            eta_list = np.linspace(best_eta / 5., best_eta * 5., num=100).tolist()

            best_acc = 0.

            if zs_classifier is None:
                clip_logits = temperature * features @ clip_weights
            else:
                clip_logits_0 = temperature * zs_classifier(features)
                attn_w = cache_keys.T @ clip_weights
                clip_logits_1 = temperature * features @ (cache_keys @ F.softmax(attn_w, dim=0)) * best_gamma
                attn_w = features @ clip_weights
                clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights * best_eta
                clip_logits = clip_logits_0 + clip_logits_1 + clip_logits_2

            paramlist = list(itertools.product(beta_list, alpha_list))
            if cache_adapter:
                affinity = features_ica @ cache_adapter(cache_keys_ica.T).T
            else:
                affinity = features_ica @ cache_keys_ica
            affinity = normalize(affinity)

            # search for alpha and beta
            for params in tqdm(paramlist):
                beta, alpha = params
                cache_logits_ica = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                cca_logits = clip_logits + cache_logits_ica * alpha
                acc = cls_acc(cca_logits, labels)
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

            if zs_classifier is None:
                clip_logits_0 = temperature * features @ clip_weights
            else:
                clip_logits_0 = temperature * zs_classifier(features)

            attn_w = cache_keys.T @ clip_weights
            clip_logits_1 = temperature * features @ (cache_keys @ F.softmax(attn_w, dim=0))
            attn_w = features @ clip_weights
            clip_logits_2 = temperature * (F.softmax(attn_w, dim=1) @ clip_weights.T) @ clip_weights

            # search for gamma and eta
            paramlist = list(itertools.product(gamma_list, eta_list))
            for params in tqdm(paramlist):
                gamma, eta = params
                clip_logits = clip_logits_0 + temperature * (clip_logits_1 * gamma + clip_logits_2 * eta)
                cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
                cca_logits = clip_logits + cache_logits * best_alpha
                acc = cls_acc(cca_logits, labels)
                if acc > best_acc:
                    best_acc = acc
                    best_gamma = gamma
                    best_eta = eta

        s = ("After searching, the best accuracy: {:.2f}, beta: {:.2f}, "
             "alpha: {:.2f}, gamma: {:.6f}, eta: {:.6f}.").format(best_acc, best_beta, best_alpha, best_gamma, best_eta)
        print(s)
        f.write(s + '\n')

    return best_beta, best_alpha, best_gamma, best_eta
