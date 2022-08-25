import pytest


class TestEgoAlloDataset:
    def test_import_make_dataset(self):
        try:
            from ego_allo_rnns.data.EgoVsAllo import make_datasets  # noqa F401
        except ImportError as e:
            return pytest.fail(e)

    def test_gen_datasets(self):
        from ego_allo_rnns.data.EgoVsAllo import make_datasets

        data = make_datasets(size_ds=20, n_train=100, n_test=100)  # noqa F481

    def test_dataset_size(self):
        from ego_allo_rnns.data.EgoVsAllo import make_datasets

        data = make_datasets(size_ds=20, n_train=100, n_test=100)
        # verify that correct number of trials
        assert len(data[0]) == 100
        assert len(data[1]) == 100

        # verify that correct number of frames
        data_train = data[0]
        x_train_samp = data_train[0][0]
        assert x_train_samp.shape[0] == 11

        # verify that correct resolution (size_ds)
        assert x_train_samp.shape[1] ** 0.5 == 20


if __name__ == "__main__":
    pytest.main([__file__])
