import glob
import os

import fire
import pandas as pd


def generate_report(report_path: str):
    summary_file = os.environ['GITHUB_STEP_SUMMARY']
    file = open(summary_file, 'w')

    subfolders = [f.path for f in os.scandir(report_path) if f.is_dir()]
    for dir_path in subfolders:
        second_subfolders = [
            f.path for f in os.scandir(dir_path) if f.is_dir()
        ]
        for sec_dir_path in second_subfolders:
            print('-' * 25, sec_dir_path.replace(report_path, ''), '-' * 25)
            file.writelines('-' * 25 + sec_dir_path.replace(report_path, '') +
                            '-' * 25 + '\n')

            benchmark_subfolders = [
                f.path for f in os.scandir(sec_dir_path) if f.is_dir()
            ]
            for benchmark_subfolder in benchmark_subfolders:
                backend_subfolders = [
                    f.path for f in os.scandir(benchmark_subfolder)
                    if f.is_dir()
                ]
                for backend_subfolder in backend_subfolders:
                    print('*' * 10,
                          backend_subfolder.replace(sec_dir_path,
                                                    ''), '*' * 10)
                    file.writelines(
                        '*' * 10 +
                        backend_subfolder.replace(sec_dir_path, '') +
                        '*' * 10 + '\n')
                    merged_csv_path = os.path.join(backend_subfolder,
                                                   'summary.csv')
                    csv_files = glob.glob(
                        os.path.join(backend_subfolder, '*.csv'))
                    average_csv_path = os.path.join(backend_subfolder,
                                                    'average.csv')
                    if merged_csv_path in csv_files:
                        csv_files.remove(merged_csv_path)
                    if average_csv_path in csv_files:
                        csv_files.remove(average_csv_path)
                    merged_df = pd.DataFrame()

                    if len(csv_files) > 0:
                        for f in csv_files:
                            df = pd.read_csv(f)
                            merged_df = pd.concat([merged_df, df],
                                                  ignore_index=True)

                        merged_df = merged_df.sort_values(
                            by=merged_df.columns[0])

                        grouped_df = merged_df.groupby(merged_df.columns[0])
                        if 'generation' not in benchmark_subfolder:
                            average_values = grouped_df.mean()
                            average_values.to_csv(average_csv_path, index=True)
                            avg_df = pd.read_csv(average_csv_path)
                            merged_df = pd.concat([merged_df, avg_df],
                                                  ignore_index=True)
                        merged_df.to_csv(merged_csv_path, index=False)
                        print(merged_df)
                        file.write(merged_df.to_string(index=False) + '\n\n')
        file.write('\n\n')

    file.close()


if __name__ == '__main__':
    fire.Fire()
