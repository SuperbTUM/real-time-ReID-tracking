class BaseDataset:
    """
    Base class of reid dataset
    """

    @staticmethod
    def get_imagedata_info(data):
        pids, cams, seqs = [], [], []
        for _, pid, camid, seqid in data:
            pids += [pid]
            cams += [camid]
            seqs += [seqid]
        seqs = set(seqs)
        num_seqs = len(seqs)
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)

        return num_pids, num_imgs, num_cams, num_seqs

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_seqs = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_seqs = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_seqs = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # sequences")
        print("  ----------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_seqs))
        print("  query    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_seqs))
        print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_seqs))
        print("  ----------------------------------------------------")

