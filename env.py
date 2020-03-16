import os
import shutil

class_names = ["circinatum",
               "garryana",
               'glabrum',
               "kelloggii",
               "macrophyllum",
               "negundo"]


def mkdir(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


# creating data sets for training and testing
def create_data_sets():
    directory = "isolated"

    pic_count = []  # count of pictures in each folder
    for subdir, dirs, files in os.walk(directory):
        pic_count.append(int(len(files)))

    folder = 0
    testi = 0
    for subdir, dirs, files in os.walk(directory):
        # variables for iteration and naming
        pici = 0
        traini = 0

        for file in os.listdir(subdir):
            if folder == 0:
                pass
            # 80% of the data (images) goes to training
            elif pici % 5 != 0:
                shutil.copyfile(subdir + "\\" + file.title(),
                                'train\\' + class_names[folder - 1] + "\\" + str(traini) + ".Jpg")
                traini = traini + 1
            # 20% of the data goes to test directory
            else:
                shutil.copyfile(subdir + "\\" + file.title(),
                                'test\\' + str(testi) + ".Jpg")
                testi = testi + 1

            pici = pici + 1
        folder = folder + 1


if __name__ == '__main__':
    mkdir("train")
    mkdir("test")
    for i in class_names:
        mkdir("train\\" + i)
    create_data_sets()
