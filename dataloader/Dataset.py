import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, num_triplets, epoch, num_human_identities_per_batch=32,
                 triplet_batch_size=544, training_triplets_path=None, transform=None):
        """
        Args:
            root_dir: Đường dẫn đến thư mục chứa dataset, mỗi thư mục con là một lớp của một người.
            num_triplets: Số lượng triplet cần được tạo.
            epoch: Số epoch hiện tại (dùng để lưu triplet đã tạo cho epoch này).
            num_human_identities_per_batch: Số lượng danh tính người trong mỗi batch.
            triplet_batch_size: Số lượng triplet trong mỗi batch.
            training_triplets_path: Đường dẫn đến file triplet numpy đã tạo từ trước, nếu có.
            transform: Thiết lập biến đổi ảnh (augmentation).
        """
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.epoch = epoch
        self.transform = transform

        self.face_classes = self.make_dictionary_for_face_class()
        self.classes = list(self.face_classes.keys())  # Danh sách các lớp (tên thư mục con)

        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets()
        else:
            print("Loading pre-generated triplets file ...")
            self.training_triplets = np.load(training_triplets_path)

    def make_dictionary_for_face_class(self):
        """
        Tạo dictionary `face_classes` theo cấu trúc:
            face_classes = {'class0': [img_path1, img_path2, ...], 'class1': [img_path1, img_path2, ...], ...}
        """
        face_classes = dict()
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                face_classes[class_name] = [os.path.join(class_path, img) for img in os.listdir(class_path)
                                            if img.endswith(('.jpg', '.png', '.jpeg'))]
        return face_classes

    def generate_triplets(self):
        triplets = []
        print("\nGenerating {} triplets ...".format(self.num_triplets))
        num_training_iterations_per_process = self.num_triplets / self.triplet_batch_size
        progress_bar = tqdm(range(int(num_training_iterations_per_process)))

        for _ in progress_bar:
            classes_per_batch = np.random.choice(self.classes, size=self.num_human_identities_per_batch, replace=False)

            for _ in range(self.triplet_batch_size):
                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while len(self.face_classes[pos_class]) < 2:  # Nếu pos_class có ít hơn 2 ảnh, chọn lại
                    pos_class = np.random.choice(classes_per_batch)
                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)

                # Chọn ngẫu nhiên Anchor và Positive từ pos_class
                if len(self.face_classes[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)
                else:
                    ianc = np.random.randint(0, len(self.face_classes[pos_class]))
                    ipos = np.random.randint(0, len(self.face_classes[pos_class]))
                    while ianc == ipos:
                        ipos = np.random.randint(0, len(self.face_classes[pos_class]))

                # Chọn ngẫu nhiên Negative từ neg_class
                ineg = np.random.randint(0, len(self.face_classes[neg_class]))

                triplets.append(
                    [
                        self.face_classes[pos_class][ianc],
                        self.face_classes[pos_class][ipos],
                        self.face_classes[neg_class][ineg],
                        pos_class,
                        neg_class
                    ]
                )
        if not os.path.exists('/kaggle/working/datasets/generated_triplets'):
            os.makedirs
        print("Saving training triplets list in 'datasets/generated_triplets' directory ...")
        np.save('/kaggle/working/datasets/generated_triplets/epoch_{}_training_triplets_{}_identities_{}_batch_{}.npy'.format(
                self.epoch, self.num_triplets, self.num_human_identities_per_batch, self.triplet_batch_size
            ),
            triplets
        )
        print("Training triplets' list Saved!\n")

        return triplets

    def __getitem__(self, idx):
        anc_img_path, pos_img_path, neg_img_path, pos_class, neg_class = self.training_triplets[idx]

        anc_img = Image.open(anc_img_path)
        pos_img = Image.open(pos_img_path)
        neg_img = Image.open(neg_img_path)

        pos_class = torch.tensor([int(pos_class)], dtype=torch.long)
        neg_class = torch.tensor([int(neg_class)], dtype=torch.long)

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)





class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, transform=None):

        super(LFWDataset, self).__init__(dir, transform)

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_lfw_paths(self, lfw_dir):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    # Modified here
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
