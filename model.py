import torch
import torch.nn as nn

class EmotionNano(nn.Module):
    def __init__(self, num_classes):
        super(EmotionNano, self).__init__()
        self.filters_fundamental_block = [11, 9, 11, 8, 11, 7, 11, 27]
        self.filters_cnn_block_1 = [27, 19, 27, 26, 27, 36]
        self.filters_cnn_block_2 = [64, 39, 64, 24, 64]

        self.fundamental_block_conv_1 = nn.Conv2d(1, self.filters_fundamental_block[0], kernel_size = 3)
        self.fundamental_block_conv_2 = nn.Conv2d(self.filters_fundamental_block[0], self.filters_fundamental_block[1], kernel_size = 3)
        self.fundamental_block_conv_3 = nn.Conv2d(self.filters_fundamental_block[1], self.filters_fundamental_block[2], kernel_size = 3)
        self.fundamental_block_conv_4 = nn.Conv2d(self.filters_fundamental_block[2], self.filters_fundamental_block[3], kernel_size = 3)
        self.fundamental_block_conv_5 = nn.Conv2d(self.filters_fundamental_block[3], self.filters_fundamental_block[4], kernel_size = 3)
        self.fundamental_block_conv_6 = nn.Conv2d(self.filters_fundamental_block[4], self.filters_fundamental_block[5], kernel_size = 3)
        self.fundamental_block_conv_7 = nn.Conv2d(self.filters_fundamental_block[5], self.filters_fundamental_block[6], kernel_size = 3)
        self.fundamental_block_conv_8 = nn.Conv2d(self.filters_fundamental_block[6], self.filters_fundamental_block[7], kernel_size = 3)
        

        self.identity_layer_1 = nn.Conv2d(27, 1, stride = (2,2), kernel_size = 3) 


        self.cnn_block_conv_1 = nn.Conv2d(self.filters_cnn_block_1[0], 3, kernel_size = 3)
        self.cnn_block_conv_2 = nn.Conv2d(self.filters_cnn_block_1[1], 3, kernel_size = 3 )
        self.cnn_block_conv_3 = nn.Conv2d(self.filters_cnn_block_1[2], 3, kernel_size = 3 )
        self.cnn_block_conv_4 = nn.Conv2d(self.filters_cnn_block_1[3], 3, kernel_size = 3 )
        self.cnn_block_conv_5 = nn.Conv2d(self.filters_cnn_block_1[4], 3, kernel_size = 3 )
        self.cnn_block_conv_6 = nn.Conv2d(self.filters_cnn_block_1[5], 3, kernel_size = 3 )


        self.identity_layer_2 = nn.Conv2d(64,1,stride=(2,2), kernel_size = 3)


        self.cnn_block_2_conv_1 = nn.Conv2d(self.filters_cnn_block_2[0], 3 , kernel_size = 3)
        self.cnn_block_2_conv_2 = nn.Conv2d(self.filters_cnn_block_2[1], 3 , kernel_size = 3)
        self.cnn_block_2_conv_3 = nn.Conv2d(self.filters_cnn_block_2[2], 3 , kernel_size = 3)
        self.cnn_block_2_conv_4 = nn.Conv2d(self.filters_cnn_block_2[3], 3 , kernel_size = 3)
        self.cnn_block_2_conv_5 = nn.Conv2d(self.filters_cnn_block_2[4], 3 , kernel_size = 3)

        self.cnn_block_2_pool_1 = nn.AvgPool2d((12,12))


        self.dense_layer = nn.Linear(in_features = 64, out_features = num_classes)


    def forward(self, x):
        x_fundamental_layer_1 = self.fundamental_block_conv_1(x)
        x_fundamental_layer_2 = self.fundamental_block_conv_2(x_fundamental_layer_1)
        x_fundamental_layer_3 = self.fundamental_block_conv_3(x_fundamental_layer_2)
        x_fundamental_layer_4 = self.fundamental_block_conv_4(torch.cat((x_fundamental_layer_1, x_fundamental_layer_3), dim = 2))
        x_fundamental_layer_5 = self.fundamental_block_conv_5(x_fundamental_layer_4)
        x_fundamental_layer_6 = self.fundamental_block_conv_6(torch.cat((x_fundamental_layer_1 , x_fundamental_layer_3 , x_fundamental_layer_5), dim = 2))
        x_fundamental_layer_7 = self.fundamental_block_conv_7(x_fundamental_layer_6)
        x_fundamental_layer_8 = self.fundamental_block_conv_8(torch.cat((x_fundamental_layer_1 , x_fundamental_layer_5 , x_fundamental_layer_7), dim = 2))


        x_identity_1 = self.identity_layer_1(torch.cat((x_fundamental_layer_1 , x_fundamental_layer_3 , x_fundamental_layer_5), dim = 2))


        x_cnn_layer_1 = self.cnn_block_conv_1(x_fundamental_layer_8)
        x_cnn_layer_2 = self.cnn_block_conv_2(torch.cat((x_cnn_layer_1 , x_identity_1), dim = 2))
        x_cnn_layer_3 = self.cnn_block_conv_3(x_cnn_layer_2)
        x_cnn_layer_4 = self.cnn_block_conv_4(torch.cat((x_cnn_layer_1 , x_cnn_layer_3), dim = 2))
        x_cnn_layer_5 = self.cnn_block_conv_5(x_cnn_layer_4)
        x_cnn_layer_6 = self.cnn_block_conv_6(torch.cat((x_cnn_layer_3 , x_cnn_layer_5 , x_cnn_layer_1), dim = 2))

        x_identity_2 = self.identity_layer_2(torch.cat((x_cnn_layer_3 , x_cnn_layer_5 , x_identity_1 , x_fundamental_layer_8), dim = 2))
        
        x_cnn_2_layer_1 = self.cnn_block_2_conv_1(x_cnn_layer_6)
        x_cnn_2_layer_2 = self.cnn_block_2_conv_2(torch.cat((x_cnn_2_layer_1 , x_identity_2), dim = 2))
        x_cnn_2_layer_3 = self.cnn_block_2_conv_3(x_cnn_2_layer_2)
        x_cnn_2_layer_4 = self.cnn_block_2_conv_4(torch.cat((x_cnn_2_layer_3 , x_cnn_2_layer_1 , x_identity_2), dim = 2))
        x_cnn_2_layer_5 = self.cnn_block_2_conv_5(x_cnn_2_layer_4)
        x_cnn_2_layer_6 = self.cnn_block_2_pool_1(torch.cat((x_cnn_2_layer_3 ,  x_cnn_2_layer_5 , x_cnn_2_layer_1 , x_identity_2), dim = 2))


        x_linear = x_cnn_2_layer_6.flatten()

        output = nn.functional.SoftMax(x_linear)

        return output


        
