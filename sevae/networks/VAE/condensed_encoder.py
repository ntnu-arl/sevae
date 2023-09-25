import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    ResNet8 architecture as encoder.
    """
    def __init__(self, input_dim, latent_dim):
        """
        Parameters:
        ----------
        input_dim: int
            Number of input channels in the image.
        latent_dim: int
            Number of latent dimensions
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.elu = nn.ELU()
        print("Defined Encoder encoder.")
    
    def define_encoder(self):
        
        # define batch norm functions
        # self.bn0 = nn.BatchNorm2d(32)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(128)

        # self.max0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # define conv functions
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv0_1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.conv0_1.bias)
        
        
        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1_1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.conv1_1.bias)

        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2_1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.conv2_1.bias)

        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2)


        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))



        



        
        # self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2)
        # nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('linear'))
        # nn.init.zeros_(self.conv3.bias)

        # self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(32, 64, kernel_size=1, stride=2)
        # nn.init.xavier_uniform_(self.conv6.weight, gain=nn.init.calculate_gain('linear'))
        # nn.init.zeros_(self.conv6.bias)

        # self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv9 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        # nn.init.xavier_uniform_(self.conv9.weight, gain=nn.init.calculate_gain('linear'))
        # nn.init.zeros_(self.conv9.bias)

        
        self.dense0 = nn.Linear(3*6*128, 512)
        self.dense1 = nn.Linear(512, 2*self.latent_dim)
        # self.dense2 = nn.Linear(2*self.latent_dim, 2*self.latent_dim)

        print("[Encoder] Encoder network initialized.")

    def forward(self, img):
        return self.encode(img)
    
    def encode(self, img):
        """
        Encodes the input image.
        """

        # conv0
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_1(x0_0)
        x0_1 = self.elu(x0_1)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)

        x0_jump_2 = self.conv0_jump_2(x0_1)

        x1_1 = x1_1 + x0_jump_2

        x1_1 = self.elu(x1_1)

        x2_0 = self.conv2_0(x1_1)
        x2_1 = self.conv2_1(x2_0)

        x1_jump3 = self.conv1_jump_3(x1_1)

        x2_1 = x2_1 + x1_jump3

        x2_1 = self.elu(x2_1)

        x3_0 = self.conv3_0(x2_1)

        x = x3_0.view(x3_0.size(0), -1)

        x = self.dense0(x)
        x = self.elu(x)
        x = self.dense1(x)
        return x





        




if __name__== '__main__':
    from torchsummary import summary
    latent_dim = 64
    encoder = Encoder(latent_dim=latent_dim, input_dim=1).to("cuda")
    summary(encoder, (1, 270, 480), device="cuda")   
