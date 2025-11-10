import csv
import cv2
old_src = "D:\\COMP9517\\project\\" # 给scv文件里候选框文件位置的前缀
dest_src = "/Users/mrcfreeman/lecture/9517/proj/9517/" #自己电脑的测试集的目录前缀

given_scvfile_path = "../valid.csv"
self_scvfile_path = "../valid.csv"

def change_src():
    with open(given_scvfile_path, 'r', newline='') as csvfile:
        lines = csvfile.readlines()
    for idx in range(len(lines)):
        lines[idx] = lines[idx].replace(old_src, dest_src)
        lines[idx] = lines[idx].replace("\\", "/")
    with open(self_scvfile_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)

change_src()
