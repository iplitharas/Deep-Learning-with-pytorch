import torch
from torchvision import models
from data_helper import DataHelper, Classifier
import os
from torch import optim, nn
import argparse


class NetworkBuilder:
    @staticmethod
    def get_pre_trained_model(network_name, hidden_layers: list, input_layer: int = 25088,
                              output: int = 102,
                              dropout_prob: float = 0.5
                              ):
        try:
            m = getattr(models, network_name.lower())(pretrained=True)
            print(f"Model:{m.__class__.__name__}:classifier is:{m.classifier})")
        except AttributeError as e:
            print("Cannot find this model in torch vision pre trained models\n"
                  "Available models: https://pytorch.org/docs/stable/torchvision/models.html\n"
                  "")
            raise e
        print("Freeze model parameters")
        for param in m.parameters():
            param.requires_grad = False
        print("Setting classifier")
        m.classifier = Classifier(input_layer=input_layer, hidden_layers=hidden_layers,output=output,
                                  dropout_prob=dropout_prob)
        print("Setting optimizer")
        optimizer = optim.Adam(m.classifier.parameters(), lr=0.001)
        return m, optimizer


class TrainNetwork(DataHelper):
    """
    Train and test the requested Neural Network and finally save it in the requested path
    """

    def __init__(self, data_path, model, learning_rate, epochs,
                 optimizer, save_dir, cuda=False, criterion=None):
        super().__init__(data_path=data_path)
        self._model = model
        self._learning_rate = learning_rate
        self._criterion = nn.NLLLoss() if not criterion else criterion
        self._epochs = epochs
        self._device = "cpu"
        self._optimizer = optimizer
        self._save_dir = save_dir
        if cuda:
            print("Asked for cuda enable,trying to enable it...")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running under --> {self._device}")

    def train_network(self):
        print_every = len(self.valid_data)
        len_testing_data = len(self.valid_data)
        print(f"Starting train and test each epoch every #{print_every}")
        steps = 0
        self._model.to(self._device)
        for e in range(self._epochs):
            print(f"Training the NN at epoch:{e + 1}/{self._epochs}")
            running_loss = 0
            for images, labels in self.train_data:
                steps += 1
                images, labels = images.to(self._device), labels.to(self._device)
                # reset the gradients
                self._optimizer.zero_grad()
                log_ps = self._model(images)
                loss = self._criterion(log_ps, labels)
                # to backward to calculate all the gradients
                loss.backward()
                # apply the new weights
                self._optimizer.step()
                # update the running loss for this epoch !
                running_loss += loss.item()
                if steps == print_every:
                    print(f"Start testing the NN at epoch:{e + 1}/{self._epochs}")
                    test_loss = 0
                    accuracy = 0
                    # turn off the gradient  and set eval to turn off the dropout
                    with torch.no_grad():
                        self._model.eval()
                        for images, labels in self.valid_data:
                            images, labels = images.to(self._device), labels.to(self._device)
                            log_ps = self._model.forward(images)
                            loss = self._criterion(log_ps, labels)
                            test_loss += loss
                            # get back the probabilities
                            ps = torch.exp(log_ps)
                            # get the top values from all the cols !
                            # need to convert them to avoid broadcasting
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                    print("Epoch: {}/{}.. ".format(e + 1, self._epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss / len_testing_data),
                          "Test Accuracy: {:.3f}%".format((accuracy / len_testing_data) * 100))

                    # reset model to train to have the drop out again
                    self._model.train()
                    running_loss = 0
                    steps = 0
        print(f"Finishing training NN at epoch: {e + 1}/{self._epochs}")
        self.test_model()
        self.save_state()
        return self._model

    def test_model(self) -> None:
        """
        Test the already trained model in the test data set
        """
        print(f"Starting Testing the model!")
        test_loss = 0
        accuracy = 0
        self._model.to(self._device)
        with torch.no_grad():
            self._model.eval()
            for images, labels in self.test_data:
                images, labels = images.to(self._device), labels.to(self._device)
                log_ps = self._model.forward(images)
                loss = self._criterion(log_ps, labels)
                test_loss += loss
                # get back the probabilities
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        print("Testing is ended")
        print("Test Loss: {:.3f}.. ".format(test_loss / len(self.test_data)),
              "Test Accuracy: {:.3f}%".format((accuracy / len(self.test_data)) * 100))

    def save_state(self) -> None:
        print(f"Saving state of model {self._model.__class__.__name__}")
        self._model.class_to_idx = self.class_to_idx
        checkpoint = {'class_to_idx': self._model.class_to_idx,
                      'criterion': self._criterion.state_dict(),
                      'optimizer': self._optimizer.state_dict(),
                      'model_state': self._model.state_dict(),
                      'model': self._model
                      }
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, self._save_dir)
            print(f"Trying to make a new directory at {output_dir}")
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                print("Output folder name:{}\nalready exists.".format(output_dir))
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.pth'))
        except Exception as e:
            print(f"Error while trying to store the checkpoint is:{e}")
        else:
            print(f"Successfully store the checkpoint at: {output_dir} ")


def create_arguments():
    parser = argparse.ArgumentParser(description='Flowers Neural Network predict')

    parser.add_argument('data_dir', action="store", default="flowers",
                        help="Path for the data flowers data set,needs also to contain"
                             " the folders train,valid,test otherwise and exception"
                             " will be thrown during parsing data")

    parser.add_argument('--hidden_layers', action="store", default=[1024, 512, 256], type=list,
                        help="Number of hidden layers must be a list"
                             " supporting len of list [1] , [2] , [3] ")

    parser.add_argument('--learning_rate', action="store", default=0.01, type=float,
                        help="learning rate as a float number for the training phase"
                             " of neural network")

    parser.add_argument('--epochs', action="store", default=10, type=int,
                        help="Number of epochs for training phase as an integer")

    parser.add_argument("--arch", action="store", default="vgg13",
                        help="Name of desired pre trained model to be used")

    parser.add_argument("--gpu", action="store_true", default=True,
                        help="if is set force all calculations to gpu")

    parser.add_argument('--save_dir', action="store", default="saved_checkpoints",
                        help="path to store the checkpoints")

    return parser.parse_args()


if __name__ == "__main__":
    arguments = create_arguments()

    model, optimizer = NetworkBuilder.get_pre_trained_model(network_name=arguments.arch,
                                                            hidden_layers=arguments.hidden_layers)
    trainer = TrainNetwork(data_path=arguments.data_dir, model=model,
                           learning_rate=arguments.learning_rate
                           , epochs=arguments.epochs,
                           optimizer=optimizer, save_dir=arguments.save_dir, cuda=arguments.gpu)
