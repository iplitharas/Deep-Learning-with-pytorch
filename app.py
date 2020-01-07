# classifier is used from the model
from predict import MakePrediction
from train import NetworkBuilder, TrainNetwork
from data_helper import DataHelper , Classifier
import sys
import cmd


class NetworkCLI(cmd.Cmd):
    intro = "Welcome to the Neural Network shell.\n" \
            "for Flowers Data set https://www.robots.ox.ac.uk/~vgg/data/flowers/" \
            "\nType help or ? to list commands.\n"
    prompt = ">>> "

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_set_helper = None
        self.prediction = None
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.learning_rate = 0.01
        self.hidden_layers = [1024, 512, 256]
        self.arch = "vgg13"
        self.data_path = None
        self.epochs = 10
        self.save_dir = "checkpoints"
        self.data_path = "flowers"
        self.top_k = 5
        self.input_layer = 25088
        self.dropout_prob = 0.5
        self.output = 102
        self.labels_path = "cat_to_name.json"

    def do_set_labels_path(self, path):
        """
        Set the path of the file containing the mapping between the classes and real-name of classes
        :param path: default value= "cat_to_name.json"
        """
        self.labels_path = path

    def do_set_output(self, output):
        """
        Set the number of classes
        :param output: integer default value = 102
        """
        self.output = int(output)

    def do_set_input_layer(self, size) -> None:
        """
        Set the input layer size (dim of features)
        :param size: an integer default value = 25088
        """
        self.input_layer = int(size)

    def do_set_dropout_prob(self, drop_out):
        """
        Set the dropout probability
        :param drop_out: float number default = 0.05
        """
        self.dropout_prob = float(drop_out)

    def do_set_data_path(self, path) -> None:
        """
        Set the path of the data to be loaded
        :param path: a string type path
        """
        self.data_path = path

    def do_set_top_k(self, top_k) -> None:
        """
        Set the number for the predictions
        :param top_k: a integer, default =5
        :return:
        """
        self.top_k = int(top_k)

    def do_set_arch(self, arch) -> None:
        """
        Set the arch of the model , default arch="vgg13"
        :param arch:
        """
        self.arch = arch

    def do_set_save_dir(self, dir) -> None:
        """
        Set the save directory for the checkpoint
        :param dir: a string type path
        """
        self.save_dir = dir
        print(f"Updating save_dir to {self.save_dir}")

    def do_set_epoch(self, epochs) -> None:
        """
        Set the number of epochs
        :param epochs: an integer , default value=10
        """
        self.epochs = int(epochs)

    def do_set_learning_rate(self, learning_rate) -> None:
        """
        Set the learning rate
        :param learning_rate: a float number , default value=0.001
        """
        self.learning_rate = float(learning_rate)

    def do_set_hidden_layers(self, line) -> None:
        """
        Set the hidden layers max 3 values: example 1024 512 256
        :param line: integers separated by empty space
        """
        line = line.split(" ")
        self.hidden_layers = []
        if len(line) > 1:
            for layer in line:
                self.hidden_layers.append(int(layer))

    def do_print_settings(self, _) -> None:
        """
        Prints all current settings
        :param _: no param
        """
        print("=" * 100)
        print("=" * 34 + "Current settings" + "=" * 50)
        print(f"Data are stored in:{self.data_path}")
        print(f"Labels mapping file name:{self.labels_path}")
        print(f"Save dir is:{self.save_dir}")
        print(f"Input dims:{self.input_layer}")
        print(f"Hidden layers: {self.hidden_layers}")
        print(f"Output #{self.output} of classes")
        print(f"Dropout prob is:{self.dropout_prob}")
        print(f"Epochs #{self.epochs}")
        print(f"Model arch is:{self.arch}")
        print(f"Checkpoint path is:{self.save_dir}")
        print(f"model is: {self.model}")
        print(f"Show top_{self.top_k} results at predictions")
        # print(f"optimizer is:{self.optimizer}")

    def do_show(self, name) -> None:
        """
        Show the train or test data set
        :param name: train or valid or test
        """
        if not self.data_set_helper:
            self.data_set_helper = DataHelper(self.data_path)
        self.data_set_helper.labels = self.labels_path
        try:
            self.data_set_helper.show_data_sets(name_of_dataset=name)
        except Exception as e:
            print(e)

    def do_build(self, _) -> None:
        """
        Build a Neural Network model and  assign this to prediction object
        :param _: no parameters
        """
        self.model, self.optimizer = \
            NetworkBuilder.get_pre_trained_model(network_name=self.arch,
                                                 input_layer=self.input_layer,
                                                 hidden_layers=self.hidden_layers,
                                                 output=self.output,
                                                 dropout_prob=self.dropout_prob)
        try:
            self.prediction = MakePrediction(self.save_dir)
            self.prediction.model = self.model
        except Exception as e:
            print(e)

    def do_train(self, _) -> None:
        """
        Train the neural network and save the state !
        always for training always try to enable cuda
        :param _:
        :return:
        """
        print(f"data path is {self.data_path}")
        print(f"learning rate is with  {self.learning_rate}")
        print(f"epochs is {self.epochs}")
        print(f"cuda is {True}")
        print(f"save directory is {self.save_dir}")
        print(f"optimizer is {self.optimizer}")
        print(f"Model classifier is {self.model.classifier}")
        try:
            self.trainer = TrainNetwork(data_path=self.data_path,
                                        model=self.model, learning_rate=self.learning_rate,
                                        epochs=self.epochs,
                                        optimizer=self.optimizer,
                                        save_dir=self.save_dir,
                                        cuda=True)

            self.trainer.train_network()
        except AttributeError as e:
            print("You need to build the model first!")

    def do_test(self, _) -> None:
        """
        Test the trained neural network
        :param _:
        """
        try:
            if not self.trainer:
                self.trainer = TrainNetwork(data_path=self.data_path,
                                            model=self.model, learning_rate=self.learning_rate,
                                            epochs=self.epochs,
                                            optimizer=self.optimizer,
                                            save_dir=self.save_dir,
                                            cuda=True)

            self.trainer.test_model()
        except AttributeError as e:
            print("You need to build or restore the model  first!")

    def do_restore(self, _) -> None:
        """
        Restore the state of the model, the prediction model is updated by default
        :param _: no params
        """
        try:
            self.prediction = MakePrediction(self.save_dir)
            self.model, criterion, self.optimizer, class_to_idx = self.prediction.load_state()

        except AttributeError as e:
            print(f"Error trying to restore from {self.save_dir} is: {e}")

    def do_predict(self, image_path) -> None:
        """
        Make a prediction in case no image path is specified a random created image will be created
        :param image_path: optional path for image path
        :return:
        """
        try:
            prob, predictions, img, label_ = \
                self.prediction.predict(image_path=image_path,
                                        labels_file_path="cat_to_name.json",
                                        topk=self.top_k)
            print("*" * 100)
            print(f"Top #{self.top_k}\n"
                  f"Predictions for --> {label_} are:\n"
                  f"Probability: {prob}\n"
                  f"Predictions: {predictions}")
            self.prediction.view_classify(img=img, ps=prob, label=label_, labels=predictions,
                                          topk=self.top_k)
        except AttributeError as e:
            print(f"You need to build or restore the model first! {e}")

    def do_exit(self, _) -> None:
        """
        Exit the program
        :return:
        """
        print("Thank you!bye")
        sys.exit()


def main() -> None:
    cli = NetworkCLI()
    cli.cmdloop()


if __name__ == "__main__":
    main()
