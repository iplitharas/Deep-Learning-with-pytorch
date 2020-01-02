from data_helper import DataHelper
import os
import torch
import argparse


class MakePrediction(DataHelper):
    """
    This class loads the state of a previous trained neural network
    and also provide method to make predictions
    """
    def __init__(self, checkpoint_path: str,
                 data_path: str = "flowers"):
        super().__init__(data_path=data_path)
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._device = "cpu"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        print("Setting model at prediction")
        self._model = model

    def load_state(self) -> tuple:
        """
        Load the previous state of the model
        """
        print("Restoring  model from checkpoint")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, self._checkpoint_path)
        print(path)
        try:
            # we consider that the save happened at cuda mode
            checkpoint = torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu")
            criterion = checkpoint['criterion']
            optimizer = checkpoint['optimizer']
            class_to_idx = checkpoint['class_to_idx']
            self._model = checkpoint['model']
            self._model.load_state_dict(checkpoint['model_state'])
        except Exception as e:
            print("The path of checkpoint is wrong")
            raise e
        return self.model, criterion, optimizer , class_to_idx

    def predict(self, image_path: str = None, topk: int = 5, cuda: bool = False,
                labels_file_path: str = None) -> tuple:
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        :return: topk probabilities, the corresponding class names if they exist,the image itself
        and the label
        """
        print(">Starting prediction")
        top_class_names = None

        if labels_file_path:
            self.labels = labels_file_path
        if cuda:
            print("Asked for cuda enable,trying to enable it...")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running under --> {self._device}")

        try:
            image, label = self.process_image(image_path)
        except (FileNotFoundError, AttributeError)as e:
            print(f"Error trying to open the image is: {e}")
            image, label = self.get_random_image()

        image = image.unsqueeze(0).float()
        self._model.to(self._device)
        # do a forward pass
        with torch.no_grad():
            self._model.eval()
            image = image.to(self._device)
            log_ps = self._model.forward(image)
            ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_class = top_class.numpy()[0]
        # check for labels
        if labels_file_path:
            mapping = self.class_to_idx
            top_class = [mapping.get(idx) for idx in top_class]
            top_class_names = [self.labels.get(str(elem)) for elem in top_class]

        # reset model to train mode
        self._model.train()
        image = image.squeeze()
        return top_p, top_class if not top_class_names else top_class_names, image, label


def create_arguments():
    parser = argparse.ArgumentParser(description='Flowers Neural Network predict')
    parser.add_argument('image_path', action='store',
                        help="Path for one image for prediction")
    parser.add_argument('checkpoint', action="store",
                        help="Path for the previous saved checkpoints")
    parser.add_argument('--category_names', action="store",
                        help="Path for the json"
                             " containing the actual mapping for each"
                             " class of images in data")
    parser.add_argument("--top_k", action="store", default=7, type=int,
                        help="Number of top 5 predictions to be calculated "
                             " from prediction ")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="if is set force all calculations to gpu")

    return parser.parse_args()


if __name__ == "__main__":
    arguments = create_arguments()
    prediction = MakePrediction(arguments.checkpoint)
    prediction.load_state()
    prob, predictions, img, label_ = prediction.predict(image_path=arguments.image_path,
                                                        topk=arguments.top_k,
                                                        labels_file_path=arguments.category_names,
                                                        cuda=arguments.gpu)
    print("*" * 100)
    print(f"Top #{arguments.top_k}\n"
          f"Predictions for --> {label_} are:\n"
          f"Probability: {prob}\n"
          f"Predictions: {predictions}")
    prediction.view_classify(img=img, ps=prob, label=label_, labels=predictions,
                             topk=arguments.top_k)
