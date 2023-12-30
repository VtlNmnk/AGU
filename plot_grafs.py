import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(folder_with_csv, path_to_csv, folder_to_save_plots, path_to_images=os.path.abspath('images_bpg')):
    full_path_to_csv = os.path.join(folder_with_csv, path_to_csv)
    df_read = pd.read_csv(full_path_to_csv, usecols=['image', 'QS', 'psnr'])
    print(os.path.exists(path_to_images))
    csvs = os.listdir(path_to_images)
    print(csvs)

    for i in range(len(csvs)):
        new_df = df_read[df_read['image'] == csvs[i]]
        new_df = new_df.reset_index(drop=True)
        print(new_df)
        lines = plt.plot(new_df.QS, new_df.psnr, label=csvs[i])

    plt.legend(framealpha=1, frameon=True)
    plt.grid()
    plt.xlabel('QS')
    plt.ylabel('psnr')
    # plt.show()
    full_path_to_plot = os.path.join(folder_to_save_plots, path_to_csv.replace(".csv", ".png"))
    plt.savefig(full_path_to_plot)
    # plt.cla()
    # plt.clf()


# path_to_csv = os.path.abspath('agu_result.csv')
# path_to_images = os.path.abspath('Images')

def run_plot_fig():
    folder_with_csv = "results"
    folder_to_save_plots = "plot_results"

    if os.path.isdir(folder_to_save_plots):
        print("{} exists. Please delete it manually!".format(folder_to_save_plots))
        exit()
    else:
        os.mkdir(folder_to_save_plots)

    results_csv_file_names = os.listdir("results")
    for path_to_csv in results_csv_file_names:
        plot_csv(folder_with_csv, path_to_csv, folder_to_save_plots)

