from turtle import hideturtle
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from system_para import number_of_classes, window_size,hidden_nodes,input_channel
from torchinfo import summary

plane1 = 32
plane2 = 64
plane3 = 128
plane4 = 256

loop = 0

class AttentionMechanism(nn.Module):
    def __init__(self,inplanes):
        super(AttentionMechanism, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.ecaconv = nn.Conv1d(1,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.fc1 = nn.Linear(in_features=inplanes, out_features=inplanes)
        self.fc2 = nn.Linear(in_features=inplanes, out_features=inplanes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.relu(self.fc1(self.avg_pool(x).view(x.size(0), 1, x.size(1))))
        max_out = self.relu(self.fc2(self.max_pool(x).view(x.size(0), 1, x.size(1))))
        # print(avg_out.shape)
        out = avg_out + max_out
        out = out.view(out.size(0), out.size(2), 1, 1)
        return self.sigmoid(out)

class AttentionMechanism_B(nn.Module):
    def __init__(self,in_planes):
        super(AttentionMechanism_B, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.ecaconv = nn.Conv1d(1,1,kernel_size=1,stride=1,padding=0,bias=False)
        
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        # avg_out = self.relu(self.ecaconv(self.avg_pool(x).view(x.size(0), 1, x.size(1))))
        # max_out = self.relu(self.ecaconv(self.max_pool(x).view(x.size(0), 1, x.size(1))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        print(avg_out.shape)
        out = avg_out + max_out
        # out = out.view(out.size(0), out.size(2), 1, 1)
        return self.sigmoid(out)

class MAM(nn.Module):
    def __init__(self,inplane1,inplane2,inplane3):
        super(MAM, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)
        self.ta = AttentionMechanism(inplane3)

    def forward(self, x):
        # print(self.ca(x).shape)
        # torch.save(self.ca(x)[0,:,0,0], 'ca_%d_%d.pt'%(loop,self.ca(x).shape[1]))
        out_ca = x * self.ca(x)
        out_ca=out_ca.permute(0,2,1,3)
        # print(self.sa(out_ca).shape)
        # torch.save(self.sa(out_ca)[0,:,0,0], 'sa_%d_%d.pt'%(loop,self.sa(out_ca).shape[1]))
        out_sa = out_ca * self.sa(out_ca)
        out_sa = out_sa.permute(0,3,2,1)
        # print(self.ta(out_sa).shape)
        # torch.save(self.ta(out_sa)[0,:,0,0], 'ta_%d_%d.pt'%(loop,self.ta(out_sa).shape[1]))
        out_ta = out_sa * self.ta(out_sa)
        out_ta = out_ta.permute(0,2,3,1)
        return out_ta

class MAM_B(nn.Module):
    def __init__(self,in_planes):
        super(MAM_B, self).__init__()
        self.ca = AttentionMechanism_B(in_planes)
        self.sa = AttentionMechanism_B(in_planes)
        self.ta = AttentionMechanism_B(in_planes)

    def forward(self, x):
        
        out_ca = x * self.ca(x)
        out_ca=out_ca.permute(0,2,1,3)
        out_sa = out_ca * self.sa(out_ca)
        out_sa = out_sa.permute(0,3,2,1)
        out_ta = out_sa * self.ta(out_sa)
        out_ta = out_ta.permute(0,2,3,1)
        return out_ta

class MAM_P(nn.Module):
    def __init__(self,inplane1,inplane2,inplane3):
        super(MAM_P, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)
        self.ta = AttentionMechanism(inplane3)

    def forward(self, x):
        out_ca = x * self.ca(x)

        out_sa = x.permute(0,2,1,3)
        out_sa = out_sa * self.sa(out_sa)
        out_sa = out_sa.permute(0,2,1,3)

        out_ta = x.permute(0,3,1,2)
        out_ta = out_ta * self.ta(out_ta)
        out_ta = out_ta.permute(0,2,3,1)

        att=1+torch.sigmoid(out_ca*out_sa*out_ta)
        return att*x

class MAM_CA(nn.Module):
    def __init__(self,inplane1,inplane2,inplane3):
        super(MAM_CA, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)
        self.ta = AttentionMechanism(inplane3)

    def forward(self, x):
        out_ca = x * self.ca(x)
        # out_ca=out_ca.permute(0,2,1,3)
        # out_sa = out_ca * self.sa(out_ca)
        # out_sa = out_sa.permute(0,3,2,1)
        # out_ta = out_sa * self.ta(out_sa)
        # out_ta = out_ta.permute(0,2,3,1)
        return out_ca
        
class MAM_SA(nn.Module):
    def __init__(self,inplane1,inplane2,inplane3):
        super(MAM_SA, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)
        self.ta = AttentionMechanism(inplane3)

    def forward(self, x):
        out_ca = x
        out_ca=out_ca.permute(0,2,1,3)
        out_sa = out_ca * self.sa(out_ca)
        out_sa = out_sa.permute(0,2,1,3)
        # out_ta = out_sa * self.ta(out_sa)
        # out_ta = out_ta.permute(0,2,3,1)
        return out_sa
class MAM_TA(nn.Module):
    def __init__(self,inplane1,inplane2,inplane3):
        super(MAM_TA, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)
        self.ta = AttentionMechanism(inplane3)

    def forward(self, x):
        out_sa = x
        # out_ca=out_ca.permute(0,2,1,3)
        # out_sa = out_ca * self.sa(out_ca)
        out_sa = out_sa.permute(0,3,2,1)
        out_ta = out_sa * self.ta(out_sa)
        out_ta = out_ta.permute(0,3,2,1)
        return out_ta
    
class CNNMAM(nn.Module):
    def __init__(self):
        super(CNNMAM, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )

        self.mam1 = MAM(plane1,input_channel,window_size)
        self.mam2 = MAM(plane2,int(input_channel/2),int(window_size/2))
        self.mam3 = MAM(plane3,int(input_channel/2),int(window_size/2))
        self.mam4 = MAM(plane4,int(input_channel/4),int(window_size/4))

        self.fc = nn.Sequential(
            # nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):

        global loop
        
        # torch.save(x[0,0,:,:], 'x_%d.pt'%loop)
        loop+=1

        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x
    
class CNNMAM_B(nn.Module):
    def __init__(self):
        super(CNNMAM_B, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )
        self.mam1 = MAM_B(plane1)
        self.mam2 = MAM_B(plane2)
        self.mam3 = MAM_B(plane3)
        self.mam4 = MAM_B(plane4)

        self.fc = nn.Sequential(
            # nn.Linear(plane3 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x
    
class CNNMAM_P(nn.Module):
    def __init__(self):
        super(CNNMAM_P, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )

        self.mam1 = MAM_P(plane1,input_channel,window_size)
        self.mam2 = MAM_P(plane2,int(input_channel/2),int(window_size/2))
        self.mam3 = MAM_P(plane3,int(input_channel/2),int(window_size/2))
        self.mam4 = MAM_P(plane4,int(input_channel/4),int(window_size/4))

        self.fc = nn.Sequential(
            # nn.Linear(plane3 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),


            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x

class CNNNOMAM(nn.Module):
    def __init__(self):
        super(CNNNOMAM, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )

        # self.mam1 = MAM()
        # self.mam2 = MAM()
        # self.mam3 = MAM()
        # self.mam4 = MAM()

        self.fc = nn.Sequential(
            # nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        # x = self.mam1(x)

        x = self.cnn2(x)
        # x = self.mam2(x)

        x = self.cnn3(x)
        # x = self.mam3(x)

        x = self.cnn4(x)
        # x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x

class CNNMAM_CA(nn.Module):
    def __init__(self):
        super(CNNMAM_CA, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )
        
        self.mam1 = MAM_CA(plane1,input_channel,window_size)
        self.mam2 = MAM_CA(plane2,int(input_channel/2),int(window_size/2))
        self.mam3 = MAM_CA(plane3,int(input_channel/2),int(window_size/2))
        self.mam4 = MAM_CA(plane4,int(input_channel/4),int(window_size/4))

        self.fc = nn.Sequential(
            # nn.Linear(plane3 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x
class CNNMAM_SA(nn.Module):
    def __init__(self):
        super(CNNMAM_SA, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )

        self.mam1 = MAM_SA(plane1,input_channel,window_size)
        self.mam2 = MAM_SA(plane2,int(input_channel/2),int(window_size/2))
        self.mam3 = MAM_SA(plane3,int(input_channel/2),int(window_size/2))
        self.mam4 = MAM_SA(plane4,int(input_channel/4),int(window_size/4))

        self.fc = nn.Sequential(
            # nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x       

class CNNMAM_TA(nn.Module):
    def __init__(self):
        super(CNNMAM_TA, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, plane1, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane1),
            nn.PReLU(plane1),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane2),
            nn.PReLU(plane2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(plane2, plane3, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(plane3),
            nn.PReLU(plane3),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(plane3, plane4, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(plane4),
            nn.PReLU(plane4),
        )
        
        self.mam1 = MAM_TA(plane1,input_channel,window_size)
        self.mam2 = MAM_TA(plane2,int(input_channel/2),int(window_size/2))
        self.mam3 = MAM_TA(plane3,int(input_channel/2),int(window_size/2))
        self.mam4 = MAM_TA(plane4,int(input_channel/4),int(window_size/4))

        self.fc = nn.Sequential(
            # nn.Linear(plane3 * int(input_channel/4) * int(window_size/4), hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, hidden_nodes),
            # nn.BatchNorm1d(hidden_nodes),
            # nn.PReLU(hidden_nodes),
            # nn.Dropout(0.3),

            # nn.Linear(hidden_nodes, number_of_classes)
            nn.Linear(plane4 * int(input_channel/4) * int(window_size/4), number_of_classes)
        )
        # self.cnncbam = self.CBAM

    def forward(self, x):
        x = self.cnn1(x)
        x = self.mam1(x)

        x = self.cnn2(x)
        x = self.mam2(x)

        x = self.cnn3(x)
        x = self.mam3(x)

        x = self.cnn4(x)
        x = self.mam4(x)

        x = self.fc(x.view(-1, plane4 * int(input_channel/4) * int(window_size/4)))
        return x

if __name__ =='__main__':
    #输出模型参数
    cbam = CNNMAM()
    
    print(summary(cbam,input_size=(1,1,input_channel,window_size)))