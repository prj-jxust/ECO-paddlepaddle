import os


data_dir = 'data/UCF-101/'

train_data = os.listdir(data_dir + 'train')
train_data = [x for x in train_data if not x.startswith('.')]
print(len(train_data))

test_data = os.listdir(data_dir + 'test')
test_data = [x for x in test_data if not x.startswith('.')]
print(len(test_data))

val_data = os.listdir(data_dir + 'val')
val_data = [x for x in val_data if not x.startswith('.')]
print(len(val_data))

f = open('data/UCF-101/train.list', 'w')
for line in train_data:
    f.write(data_dir + 'train/' + line + '\n')
f.close()
f = open('data/UCF-101/test.list', 'w')
for line in test_data:
    f.write(data_dir + 'test/' + line + '\n')
f.close()
f = open('data/UCF-101/val.list', 'w')
for line in val_data:
    f.write(data_dir + 'val/' + line + '\n')
f.close()

