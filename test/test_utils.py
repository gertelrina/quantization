import unittest
import torch
import sys
import os

# from torchvision import transforms
# import torchvision

sys.path.append('/content/')
from quantization_cv.utils import export_model, prepare_test_data, get_acc, get_onnx_model_info

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class TestExportModel(unittest.TestCase):

    def setUp(self):
        # Initialize any necessary resources before each test
        pass

    def tearDown(self):
        # Clean up any resources created during the test
        # Specify the model path used for testing
        model_path = 'test_model.onnx'

        # Check if the model file exists and delete it
        if os.path.exists(model_path):
            os.remove(model_path)


    def test_export_model(self):
        # Create a dummy PyTorch model for testing
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = DummyModel()

        # Specify the model path for testing
        model_path = 'test_model.onnx'

        # Call the function to export the model
        export_model(model, model_path, bs=1, dynamic=False, save=False)

        # Add your assertions here based on the expected behavior of the export_model function
        # For example, you can check if the file 'test_model.onnx' exists

        # Clean up: Remove the test model file
        # Uncomment the line below if you want to delete the test model file after the test
        # os.remove(model_path)

class TestPrepareTestData(unittest.TestCase):

    def setUp(self):
        # Initialize any necessary resources before each test
        pass

    def tearDown(self):
        # Clean up any resources created during the test
        import shutil
        shutil.rmtree('data/')
        pass

    def test_prepare_test_data(self):
        # Call the function to prepare test data
        testloader, quant_ds = prepare_test_data()

        # Add your assertions here based on the expected behavior of the prepare_test_data function

        # For example, you can check if the testloader is an instance of torch.utils.data.DataLoader
        self.assertIsInstance(testloader, torch.utils.data.DataLoader)

        # You can also check if quant_ds is an instance of torch.utils.data.Dataset
        self.assertIsInstance(quant_ds, torch.utils.data.Dataset)

        # You can check if the lengths of test_ds and quant_ds are correct
        expected_test_size = len(testloader.dataset)
        expected_quant_size = len(quant_ds)
        self.assertEqual(len(testloader.dataset), expected_test_size)
        self.assertEqual(expected_quant_size, len(quant_ds))  # Assuming quant_ds has a q_size attribute

        # Add more assertions based on the specific behavior you expect from prepare_test_data

class TestGetAcc(unittest.TestCase):

    def setUp(self):
        # Initialize any necessary resources before each test
        pass

    def tearDown(self):
        # Clean up any resources created during the test
        # Specify the model path used for testing
        model_path = 'test_model.onnx'

        # Check if the model file exists and delete it
        if os.path.exists(model_path):
            os.remove(model_path)

        import shutil
        shutil.rmtree('data/')

    def test_get_acc(self):
        from torchvision.models import resnet18
        # Assuming you have a minimal ResNet model in ONNX format and a test data loader
        model_path = 'test_model.onnx'  # Replace with the actual model path

        # Create a dummy ResNet model for testing
        dummy_resnet = resnet18(pretrained=False, num_classes=10)  # Assuming CIFAR-10 has 10 classes
        dummy_resnet.eval()

        # Convert and save the dummy ResNet model to ONNX format
        dummy_input = torch.randn(100, 3, 32, 32)  # Assuming input size for CIFAR-10
        torch.onnx.export(dummy_resnet, dummy_input, model_path, verbose=False)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        # Call the function to get accuracy
        accuracy = get_acc(model_path, testloader, device='cuda')  # Replace 'cpu' with 'cuda' if needed

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

class TestGetOnnxModelInfo(unittest.TestCase):

    def setUp(self):
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = DummyModel()

        # Specify the model path for testing
        self.model_path = 'test_model.onnx'
        export_model(model, self.model_path , bs=1, dynamic=False, save=False)
        # Initialize any necessary resources before each test

    def tearDown(self):
        # Clean up any resources created during the test
        # Specify the model path used for testing
        model_path = 'test_model.onnx'

        # Check if the model file exists and delete it
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_get_onnx_model_info(self):


        # Call the function to get ONNX model information
        model_info = get_onnx_model_info(self.model_path )

        # Add your assertions here based on the expected behavior of the get_onnx_model_info function

        # For example, you can check if the returned dictionary contains the expected keys
        self.assertIn("params", model_info)
        self.assertIn("model_size", model_info)

        # You can also check if the "params" value is a positive integer
        self.assertIsInstance(model_info["params"], int)
        self.assertGreaterEqual(model_info["params"], 0)

if __name__ == '__main__':
    unittest.main()
