from code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        data = self.dataset.load()

        self.method.data = {
            'train': data['train'],
            'test': data['test']
        }

        result = self.method.run()

        self.result.data = result
        self.result.save()

        self.evaluate.data = result

        return self.evaluate.evaluate(), None


if __name__ == "__main__":

    from code.stage_2_code.Dataset_Loader import Dataset_Loader
    from code.stage_2_code.Method_MLP import Method_MLP
    from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
    from code.stage_2_code.Result_Loader import Result_Loader

    dataset = Dataset_Loader()
    dataset.dataset_source_folder_path = "./code/stage_2_code/data/"

    temp_data = dataset.load()

    input_dim = len(temp_data['train']['X'][0])
    num_classes = len(set(temp_data['train']['y']))

    method = Method_MLP(
        "MLP",
        "stage2",
        input_dim=input_dim,
        num_classes=num_classes
    )

    evaluate = Evaluate_Accuracy()
    result = Result_Loader()

    setting = Setting_Train_Test_Split()
    setting.dataset = dataset
    setting.method = method
    setting.evaluate = evaluate
    setting.result = result

    setting.load_run_save_evaluate()