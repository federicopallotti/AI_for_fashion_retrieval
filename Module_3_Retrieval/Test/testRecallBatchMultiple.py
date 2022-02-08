import os
from models.GarnmentsNet import GarnmentsMobileNet
import matplotlib.pyplot as plt
import torch
from utils import l2norm
from ComposeDataset import FeedDatasetTestingRecall
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import platform
from neptuneLogger import NetpuneLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess_for_items = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_for_parsed = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def test(category_test, total_it, train_parameters, hyper, start):
    category = hyper["category"]
    print("Starting...")
    if platform.system() == 'Windows':
        dataset_path = 'C:\\Users\\pales\\Desktop\\DatasetCV'
        annotation_file = dataset_path + '\\' + category + '\\test_pairs_paired.txt'
        root_dir = 'C:\\Users\\pales\\Desktop\\DatasetCV\\' + category + '\\images'
        weights_path = "data\\model-weights.pth"
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        dataset_path = '../DatasetFolder/DatasetCV'
        annotation_file = dataset_path + '/' + category + '/test_pairs_paired.txt'
        root_dir = dataset_path + '/' + category + '/images'
        weights_path = "data/weights-pytorch-lower_bodySun Aug  8 18:50:19 2021.pth"

    model = GarnmentsMobileNet().to(device)

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    neptune = NetpuneLogger(category_test, train_parameters, hyper)
    score_recall20 = 0
    score_recall10 = 0
    score_recall5 = 0
    score_recall1 = 0
    for j in range(total_it):
        img0, ground = FeedDatasetTestingRecall.get(annotations_file=annotation_file, idx=j + start)

        # Taking image #########
        path_of_the_query_img = img0  # il percorso dell'immagine query dalla root_dir
        path_os = path_of_the_query_img.split('/')  # per windows, va bene anche per linux
        path_real = os.path.join(root_dir, path_os[0], path_os[1])
        # Taking image #########
        testing_data = FeedDatasetTestingRecall(annotations_file=annotation_file,
                                                root_dir=root_dir, transform_items=preprocess_for_items,
                                                category=img0.split('/')[0])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        mean = torch.tensor(mean)
        std = torch.tensor(std)

        std.resize_(1, 1, 3)
        mean.resize_(1, 1, 3)

        test_loader = DataLoader(
            testing_data, batch_size=hyper["batch_size"], shuffle=True, num_workers=hyper["num_workers"]
        )

        matches = torch.ones((1, 224, 224, 3))
        matches_scores = torch.ones((1,))
        matches_items = []

        es = plt.imread(path_real)

        # plt.imshow(es)
        # plt.show()

        es = torch.from_numpy(es.copy()).permute(2, 0, 1).type(torch.float32)
        es = preprocess_for_parsed(es)
        es = es.to(device)
        es = es.unsqueeze(0)

        for batch, img0, img1 in tqdm(test_loader):
            x = batch
            x = x.to(device)

            out1, out2 = model.forward(es, x)
            out1 = l2norm(out1, dim=-1).cpu()
            out2 = l2norm(out2, dim=-1).cpu()
            score = torch.mm(out1, out2.t())

            score = score.squeeze(0)

            score = score.detach().cpu()

            vis = (x.detach().cpu().permute(0, 2, 3, 1) * std + mean).type(torch.uint8)

            matches = torch.cat((matches, vis), 0)

            matches_scores = torch.cat((matches_scores, score), 0)

            matches_items.extend(list(img1))

        matches = matches[1:].numpy()
        matches_scores = matches_scores[1:]
        matches_scores = torch.flatten(matches_scores, 0).numpy()

        # print('matches score before ordering : ', matches_scores)
        matches_scores = np.argsort(matches_scores)[::-1]
        # print('matches score after ordering : ', matches_scores)

        recall20 = 20
        recall10 = 10
        recall5 = 5
        recall1 = 1

        # CODE FOR VISUALIZING

        # for i in range(recall):
        #     if (i == len(matches)):
        #         break
        #     index = matches_scores[i]
        #     im = matches[index]
        #     im = im.astype(np.uint8)
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # #     cv2.imshow("Immagine Output", im)
        # #     if cv2.waitKey(0) == ord('q'):
        # #         cv2.destroyAllWindows()
        # #         break
        # #     # plt.imshow(im)
        # #     # plt.show()
        # # cv2.destroyAllWindows()

        verdict = "YES" if ground in matches_items else "NO"
        ordered_items = np.array(matches_items)
        ordered_items = ordered_items[matches_scores]
        ordered_items = list(ordered_items)

        if verdict == "YES":
            item_index = ordered_items.index(ground)

        if verdict == "YES" and item_index < recall20:
            print(f"ground is in recall 20 ? : ", verdict,
                  f", with a precision of {item_index + 1} over {len(testing_data)} data")
            score_recall20 += 1
        else:
            print(f"ground is in recall 20 ? : ", "NO")
            print('The total number of image to compare was : ', len(testing_data))

        if verdict == "YES" and item_index < recall10:
            score_recall10 += 1
        if verdict == "YES" and item_index < recall5:
            score_recall5 += 1
        if verdict == "YES" and item_index < recall1:
            score_recall1 += 1

        neptune.set(score_recall20, j + 1, recall20)
        neptune.set(score_recall10, j + 1, recall10)
        neptune.set(score_recall5, j + 1, recall5)
        neptune.set(score_recall1, j + 1, recall1)

    print(f"accuracy for recall 20 is : ", score_recall20 / total_it)
    print(f"accuracy for recall 10 is : ", score_recall10 / total_it)
    print(f"accuracy for recall 5 is : ", score_recall5 / total_it)
    print(f"accuracy for recall 1 is : ", score_recall1 / total_it)
    neptune.update(score_recall20 / total_it, recall20)
    neptune.update(score_recall10 / total_it, recall10)
    neptune.update(score_recall5 / total_it, recall5)
    neptune.update(score_recall1 / total_it, recall1)
    neptune.destroy()
