import os

path = os.getcwd()

month = 'June'

for i in range(2012, 2021):
    new_folder_path = 'downloads/new'

    new_folder_path = f'{new_folder_path}/{str(i)}/{month}'
    new_path = os.path.join(path, new_folder_path)
    print(new_path)

    try:
        os.makedirs(f'downloads_new/{i}/{month}')
    except:
        print("Can't make!")