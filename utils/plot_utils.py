from tqdm import trange
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime


# def gather_features(model, testset):
#     gathered_token_classes = []
#     gathered_token_features = []
#     gathered_token_features_0 = []
#     gathered_token_features_1 = []
#     gathered_token_features_2 = []
#     gathered_token_features_3 = []
#     gathered_token_features_4 = []

#     for j in trange(testset.batch_count):
#         sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = testset.get_batch(j)
#         with torch.no_grad():
#             features = model.bert(bert_tokens, masks)
#             # print(features['last_hidden_state'].shape)
#             features = features['last_hidden_state']

#         gathered_token_features.append(features)
#         gathered_token_classes.extend(token_classes)

#     for features, token_classes in zip(gathered_token_features, gathered_token_classes):
        
#         # print(features.shape, token_classes)
#         L = len(token_classes)
#         # for i in range(len(token_classes)):
#         #     token = features[:, i, :]   
#         #     print(i, token_classes[i])
#         #     print(features[:, token, :].shape)
#         # print(token_classes)
#         # print(L)
#         features = features[:, :L, :]
#         token_classes = np.array(token_classes)
        
#         # print(features.shape)
#         # print([token_classes == 0])
#         token_class_0 = features[:, token_classes == 0, :]
#         # print(token_class_0.shape)
#         token_class_1 = features[:, token_classes == 1, :]
#         token_class_2 = features[:, token_classes == 2, :]
#         token_class_3 = features[:, token_classes == 3, :]
#         token_class_4 = features[:, token_classes == 4, :]

#         gathered_token_features_0.extend(token_class_0)
#         gathered_token_features_1.extend(token_class_1)
#         gathered_token_features_2.extend(token_class_2)
#         gathered_token_features_3.extend(token_class_3)
#         gathered_token_features_4.extend(token_class_4)


#     def filter_tensors(tensor_list):
#         return [t for t in tensor_list if all(dim > 0 for dim in t.shape)]

#     gathered_token_features_0 = filter_tensors(gathered_token_features_0)
#     gathered_token_features_1 = filter_tensors(gathered_token_features_1)
#     gathered_token_features_2 = filter_tensors(gathered_token_features_2)
#     gathered_token_features_3 = filter_tensors(gathered_token_features_3)
#     gathered_token_features_4 = filter_tensors(gathered_token_features_4)

#     gathered_token_features_0 = torch.cat(gathered_token_features_0, dim=0).cpu().numpy()
#     gathered_token_features_1 = torch.cat(gathered_token_features_1, dim=0).cpu().numpy()
#     gathered_token_features_2 = torch.cat(gathered_token_features_2, dim=0).cpu().numpy()
#     gathered_token_features_3 = torch.cat(gathered_token_features_3, dim=0).cpu().numpy()
#     gathered_token_features_4 = torch.cat(gathered_token_features_4, dim=0).cpu().numpy()

#     gathered_token_features_0.shape, gathered_token_features_1.shape, gathered_token_features_2.shape, gathered_token_features_3.shape, gathered_token_features_4.shape

#     return gathered_token_features_0, gathered_token_features_1, gathered_token_features_2, gathered_token_features_3, gathered_token_features_4


# # def gather_features(model, testset):
# #     gathered_token_classes = []
# #     gathered_token_features = []
# #     gathered_token_features_0 = []
# #     gathered_token_features_1 = []
# #     gathered_token_features_2 = []
# #     gathered_token_features_3 = []
# #     gathered_token_features_4 = []

# #     for j in trange(testset.batch_count):
# #         sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = testset.get_batch(j)
# #         with torch.no_grad():
# #             features = model.bert(bert_tokens, masks)
# #             # print(features['last_hidden_state'].shape)
# #             features = features['last_hidden_state']

# #         gathered_token_features.append(features)
# #         gathered_token_classes.extend(token_classes)

# #     # gathered_token_features = torch.cat(gathered_token_features, dim=0)
# #     # print(gathered_token_features)

# #     for features, token_classes in zip(gathered_token_features, gathered_token_classes):
# #         # print(features.shape, token_classes)
# #         L = len(token_classes)
# #         # print(L)
# #         features = features[:, :L, :]
# #         token_classes = np.array(token_classes)
        
# #         # print(features.shape)
# #         # print([token_classes == 0])
# #         token_class_0 = features[:, token_classes == 0, :]
# #         # print(token_class_0.shape)
# #         token_class_1 = features[:, token_classes == 1, :]
# #         token_class_2 = features[:, token_classes == 2, :]
# #         token_class_3 = features[:, token_classes == 3, :]
# #         token_class_4 = features[:, token_classes == 4, :]

# #         gathered_token_features_0.extend(token_class_0)
# #         gathered_token_features_1.extend(token_class_1)
# #         gathered_token_features_2.extend(token_class_2)
# #         gathered_token_features_3.extend(token_class_3)
# #         gathered_token_features_4.extend(token_class_4)

# #     gathered_token_features_0 = torch.cat(gathered_token_features_0, dim=0).cpu().numpy()
# #     gathered_token_features_1 = torch.cat(gathered_token_features_1, dim=0).cpu().numpy()
# #     gathered_token_features_2 = torch.cat(gathered_token_features_2, dim=0).cpu().numpy()
# #     gathered_token_features_3 = torch.cat(gathered_token_features_3, dim=0).cpu().numpy()
# #     gathered_token_features_4 = torch.cat(gathered_token_features_4, dim=0).cpu().numpy()

# #     return gathered_token_features_0, gathered_token_features_1, gathered_token_features_2, gathered_token_features_3, gathered_token_features_4

# # def plot_pca_3d(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4):
# #     # random sample N points for each class, where N is the number of points in the smallest class
# #     N = 6 * min(gathered_token_class_0.shape[0], 
# #             gathered_token_class_1.shape[0], 
# #             gathered_token_class_2.shape[0], 
# #             gathered_token_class_3.shape[0], 
# #             gathered_token_class_4.shape[0])
    
# #     gathered_token_class_0 = gathered_token_class_0[np.random
# #         .choice(gathered_token_class_0.shape[0], min(N, gathered_token_class_0.shape[0]), replace=False), :]
# #     gathered_token_class_1 = gathered_token_class_1[np.random
# #         .choice(gathered_token_class_1.shape[0], min(N, gathered_token_class_1.shape[0]), replace=False), :]
# #     gathered_token_class_2 = gathered_token_class_2[np.random
# #         .choice(gathered_token_class_2.shape[0], min(N, gathered_token_class_2.shape[0]), replace=False), :]
# #     gathered_token_class_3 = gathered_token_class_3[np.random
# #         .choice(gathered_token_class_3.shape[0], min(N, gathered_token_class_3.shape[0]), replace=False), :]
# #     gathered_token_class_4 = gathered_token_class_4[np.random
# #         .choice(gathered_token_class_4.shape[0], min(N, gathered_token_class_4.shape[0]), replace=False), :]

    
# #     pca = PCA(n_components=3)

# #     pca.fit(gathered_token_class_0)
# #     pca_0 = pca.transform(gathered_token_class_0)

# #     pca.fit(gathered_token_class_1)
# #     pca_1 = pca.transform(gathered_token_class_1)

# #     pca.fit(gathered_token_class_2)
# #     pca_2 = pca.transform(gathered_token_class_2)

# #     pca.fit(gathered_token_class_3)
# #     pca_3 = pca.transform(gathered_token_class_3)

# #     pca.fit(gathered_token_class_4)
# #     pca_4 = pca.transform(gathered_token_class_4)


# #     fig = plt.figure()
# #     ax = fig.add_subplot(111, projection='3d')
# #     ax.scatter(pca_0[:,0], pca_0[:,1], pca_0[:,2], c='r', label='NULL')
# #     ax.scatter(pca_1[:,0], pca_1[:,1], pca_1[:,2], c='g', label='Aspect')
# #     ax.scatter(pca_2[:,0], pca_2[:,1], pca_2[:,2], c='b', label='Opinion-POS')
# #     ax.scatter(pca_3[:,0], pca_3[:,1], pca_3[:,2], c='y', label='Opinion-NEU')
# #     ax.scatter(pca_4[:,0], pca_4[:,1], pca_4[:,2], c='m', label='Opinion-NEG')
# #     plt.legend()
# #     plt.show()


def gather_features(model, testset):
    gathered_token_classes = []
    gathered_token_features = []
    gathered_token_features_0 = []
    gathered_token_features_1 = []
    gathered_token_features_2 = []
    gathered_token_features_3 = []
    gathered_token_features_4 = []

    for j in trange(testset.batch_count):
        sentence_ids, bert_tokens, masks, word_spans, tagging_matrices, tokenized, cl_masks, token_classes = testset.get_batch(j)
        with torch.no_grad():
            features = model.bert(bert_tokens, masks)
            # print(features['last_hidden_state'].shape)
            features = features['last_hidden_state']

        gathered_token_features.append(features)
        # print(token_classes[0])
        gathered_token_classes.extend(token_classes)

        # print(len(token_classes), len(features))

        # print(len(token_classes), len(features))
        # token_classes


        # 示例
        # nested_list = [[1, 2], [3, 4, 5], [6, 7, 8]]
        # gathered_token_features = flatten(gathered_token_features)
        # print(flat_list)
    gathered_token_features = torch.cat(gathered_token_features, dim=0)
    print(len(gathered_token_classes), len(gathered_token_features))   

    for features, token_classes in zip(gathered_token_features, gathered_token_classes):
        
        L = len(token_classes)
        # for i in range(len(token_classes)):
        #     token = features[:, i, :]   
        #     print(i, token_classes[i])
        #     print(features[:, token, :].shape)
        # print(token_classes)
        # print(L)
        features = features[:L, :]
        token_classes = np.array(token_classes)
        # print(features.shape, token_classes)

        for token, token_class in zip(features, token_classes):
            # print(token_class)
            if token_class == 0:
                gathered_token_features_0.append(token)
            elif token_class == 1:
                gathered_token_features_1.append(token)
            elif token_class == 2:
                gathered_token_features_2.append(token)
            elif token_class == 3:
                gathered_token_features_3.append(token)
            elif token_class == 4:
                gathered_token_features_4.append(token)
            else:
                print('error')
            #

        # break
        # print(features.shape)
        # print([token_classes == 0])
        # token_class_0 = features[:, token_classes == 0, :]
        # # print(token_class_0.shape)
        # token_class_1 = features[:, token_classes == 1, :]
        # token_class_2 = features[:, token_classes == 2, :]
        # token_class_3 = features[:, token_classes == 3, :]
        # token_class_4 = features[:, token_classes == 4, :]

        # gathered_token_features_0.extend(token_class_0)
        # gathered_token_features_1.extend(token_class_1)
        # gathered_token_features_2.extend(token_class_2)
        # gathered_token_features_3.extend(token_class_3)
        # gathered_token_features_4.extend(token_class_4)


    # def filter_tensors(tensor_list):
    #     return [t for t in tensor_list if all(dim > 0 for dim in t.shape)]

    # gathered_token_features_0 = filter_tensors(gathered_token_features_0)
    # gathered_token_features_1 = filter_tensors(gathered_token_features_1)
    # gathered_token_features_2 = filter_tensors(gathered_token_features_2)
    # gathered_token_features_3 = filter_tensors(gathered_token_features_3)
    # gathered_token_features_4 = filter_tensors(gathered_token_features_4)

    # gathered_token_features_0 = torch.cat(gathered_token_features_0, dim=0).cpu().numpy()
    # gathered_token_features_1 = torch.cat(gathered_token_features_1, dim=0).cpu().numpy()
    # gathered_token_features_2 = torch.cat(gathered_token_features_2, dim=0).cpu().numpy()
    # gathered_token_features_3 = torch.cat(gathered_token_features_3, dim=0).cpu().numpy()
    # gathered_token_features_4 = torch.cat(gathered_token_features_4, dim=0).cpu().numpy()

    # print(gathered_token_features_0.shape, gathered_token_features_1.shape, gathered_token_features_2.shape, gathered_token_features_3.shape, gathered_token_features_4.shape)
    # print(len(gathered_token_features_0), len(gathered_token_features_1), len(gathered_token_features_2), len(gathered_token_features_3), len(gathered_token_features_4))

# # gathered_token_features = torch.cat(gathered_token_features, dim=0)
# # print(gathered_token_features)

# for features, token_classes in zip(gathered_token_features, gathered_token_classes):
#     # print(features.shape, token_classes)
#     L = len(token_classes)
#     print(token_classes)
#     # print(L)
#     features = features[:, :L, :]
#     token_classes = np.array(token_classes)
    
#     # print(features.shape)
#     # print([token_classes == 0])
#     token_class_0 = features[:, token_classes == 0, :]
#     # print(token_class_0.shape)
#     token_class_1 = features[:, token_classes == 1, :]
#     token_class_2 = features[:, token_classes == 2, :]
#     token_class_3 = features[:, token_classes == 3, :]
#     token_class_4 = features[:, token_classes == 4, :]

#     gathered_token_features_0.extend(token_class_0)
#     gathered_token_features_1.extend(token_class_1)
#     gathered_token_features_2.extend(token_class_2)
#     gathered_token_features_3.extend(token_class_3)
#     gathered_token_features_4.extend(token_class_4)

#     break
# # print(len(gathered_token_features_3))

# # gathered_token_features_0 = torch.cat(gathered_token_features_0, dim=0).cpu().numpy()
# # gathered_token_features_1 = torch.cat(gathered_token_features_1, dim=0).cpu().numpy()
# # gathered_token_features_2 = torch.cat(gathered_token_features_2, dim=0).cpu().numpy()
# # gathered_token_features_3 = torch.cat(gathered_token_features_3, dim=0).cpu().numpy()
# # gathered_token_features_4 = torch.cat(gathered_token_features_4, dim=0).cpu().numpy()
    gathered_token_features_0 = torch.stack(gathered_token_features_0).cpu().numpy()
    gathered_token_features_1 = torch.stack(gathered_token_features_1).cpu().numpy()
    gathered_token_features_2 = torch.stack(gathered_token_features_2).cpu().numpy()
    gathered_token_features_3 = torch.stack(gathered_token_features_3).cpu().numpy()
    gathered_token_features_4 = torch.stack(gathered_token_features_4).cpu().numpy()

    return gathered_token_features_0, gathered_token_features_1, gathered_token_features_2, gathered_token_features_3, gathered_token_features_4


def plot_pca(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, epoch):

    # random sample N points for each class, where N is the number of points in the smallest class
    gather = [gathered_token_class_0, 
            gathered_token_class_1, 
            gathered_token_class_2, 
            gathered_token_class_3, 
            gathered_token_class_4]
    c_s = ['r', 'g', 'b', 'y', 'm']
    labels = ['NULL', 'Aspect', 'Opinion-POS', 'Opinion-NEU', 'Opinion-NEG']

    gather_ = [i for i in gather if i.shape[0] != 0]
    gather_n = [i.shape[0] for i in gather_]
    c_s_ = [c_s[i] for i in range(len(gather)) if gather[i].shape[0] != 0]
    labels_ = [labels[i] for i in range(len(gather)) if gather[i].shape[0] != 0]

    N = 6 * min(gather_n)
    
    pca = PCA(n_components=2)

    # pca.fit(gathered_token_class_0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # print(len(gather_), len(c_s_), len(labels_), N)

    for i, c, label in zip(gather_, c_s_, labels_):
        i = i[np.random.choice(i.shape[0], min(N, i.shape[0]), replace=False), :]
        pca.fit(i)
        pca_i = pca.transform(i)
        ax.scatter(pca_i[:,0], pca_i[:,1], c=c, label=label)

    plt.legend()
    plt.savefig(f'./plots_saved/pca_2d_{datetime.datetime.today()}_{epoch}.png')

 
def plot_pca_3d(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, epoch):
    # random sample N points for each class, where N is the number of points in the smallest class
    gather = [gathered_token_class_0, 
            gathered_token_class_1, 
            gathered_token_class_2, 
            gathered_token_class_3, 
            gathered_token_class_4]
    c_s = ['r', 'g', 'b', 'y', 'm']
    labels = ['NULL', 'Aspect', 'Opinion-POS', 'Opinion-NEU', 'Opinion-NEG']

    gather_ = [i for i in gather if i.shape[0] != 0]
    gather_n = [i.shape[0] for i in gather_]
    c_s_ = [c_s[i] for i in range(len(gather)) if gather[i].shape[0] != 0]
    labels_ = [labels[i] for i in range(len(gather)) if gather[i].shape[0] != 0]

    N = 6 * min(gather_n)
    
    pca = PCA(n_components=3)

    # pca.fit(gathered_token_class_0)

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # print(len(gather_), len(c_s_), len(labels_), N)

    for i, c, label in zip(gather_, c_s_, labels_):
        i = i[np.random.choice(i.shape[0], min(N, i.shape[0]), replace=False), :]
        pca.fit(i)
        pca_i = pca.transform(i)
        ax.scatter(pca_i[:,0], pca_i[:,1], pca_i[:,2], c=c, label=label)

    plt.legend()
    plt.savefig(f'./plots_saved/pca_3d_{datetime.datetime.today()}_{epoch}.png')


def plot_pca_3d_rand(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, epoch):
    # random sample N points for each class, where N is the number of points in the smallest class
    gather = [gathered_token_class_0, 
            gathered_token_class_1, 
            gathered_token_class_2, 
            gathered_token_class_3, 
            gathered_token_class_4]
    c_s = ['r', 'g', 'b', 'y', 'm']
    labels = ['NULL', 'Aspect', 'Opinion-POS', 'Opinion-NEU', 'Opinion-NEG']

    gather_ = [i for i in gather if i.shape[0] != 0]
    gather_n = [i.shape[0] for i in gather_]
    c_s_ = [c_s[i] for i in range(len(gather)) if gather[i].shape[0] != 0]
    labels_ = [labels[i] for i in range(len(gather)) if gather[i].shape[0] != 0]

    N = 6 * min(gather_n)
    
    pca = PCA(n_components=3)

    # pca.fit(gathered_token_class_0)

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # # print(len(gather_), len(c_s_), len(labels_), N)
    # for i in range(N):


    for i, c, label in zip(gather_, c_s_, labels_):
        i = i[np.random.choice(i.shape[0], min(N, i.shape[0]), replace=False), :]
        pca.fit(i)
        pca_i = pca.transform(i)
        ax.scatter(pca_i[:,0], pca_i[:,1], pca_i[:,2], c=c, label=label)

    plt.legend()
    plt.savefig(f'./plots_saved/pca_3d_{datetime.datetime.today()}_{epoch}.png')


def plot_pca_3d_rand(gathered_token_class_0, gathered_token_class_1, gathered_token_class_2, gathered_token_class_3, gathered_token_class_4, epoch):
    # stack the gathered tokens
    gather = [gathered_token_class_0, 
            gathered_token_class_1, 
            gathered_token_class_2, 
            gathered_token_class_3, 
            gathered_token_class_4]
    len_gather = [i.shape[0] for i in gather]
    min_len = min(len_gather)
    max_sampling_points = 4 * min_len
    gather = [it[np.random.choice(it.shape[0], min(max_sampling_points, len_gather[i]), replace=False), :] for i, it in enumerate(gather)]

    len_gather = [i.shape[0] for i in gather]
    c_s = ['r', 'g', 'b', 'y', 'm']
    labels = ['NULL', 'Aspect', 'Opinion-POS', 'Opinion-NEU', 'Opinion-NEG']

    gather_classes = []
    gather_labels = []
    for i in range(len(c_s)):
        gather_classes.extend([c_s[i]] * len_gather[i])
        gather_labels.extend([labels[i]] * len_gather[i])
    # print(gather_classes, gather_labels)
    print(len(gather_classes), len(gather_labels))  
    # random shuffle
    indices = list(range(sum(len_gather)))
    random.shuffle(indices)
    gather_classes = np.array([gather_classes[i] for i in indices])
    gather_labels = np.array([gather_labels[i] for i in indices])
    # print(len(gather))

    gather = np.concatenate(gather, axis=0)
    print(gather.shape)
    gather_shuffled = gather[indices, :]
    print(gather_shuffled.shape)

    pca = PCA(n_components=3)
    pca.fit(gather_shuffled)
    pca_gather = pca.transform(gather_shuffled)
    
    print(pca_gather.shape)

    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pca_gather[:,0], pca_gather[:,1], pca_gather[:,2], c=gather_classes, label=gather_labels)

    # axis label formatting, round to 0 decimal places
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c_s[i], markersize=10, label=labels[i]) for i in range(len(labels))]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig(f'./plots_saved/pca_3d_{datetime.datetime.today()}_{epoch}.png')
    # plt.show()
