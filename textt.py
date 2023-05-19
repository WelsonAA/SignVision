import cv2
from matplotlib import pyplot as plt

x=str(input())
arr=[*x]
print(arr)
for i in range(len(arr)):
    if(arr[i]==" "):
        frame=cv2.imread(f"asl/space.jpg")
    else:
        frame=cv2.imread(f"asl/{arr[i]}.jpeg")
    plt.title(arr[i])
    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.pause(1)
