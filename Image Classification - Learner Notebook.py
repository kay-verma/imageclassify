#!/usr/bin/env python
# coding: utf-8

# # Download the Data

# In[50]:


get_ipython().system('pip3 install tqdm')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import tarfile
import urllib
import shutil
import json
import random
import numpy as np
import boto3
import sagemaker

from tqdm import tqdm
from sagemaker.amazon.amazon_estimator import get_image_uri
from matplotlib import pyplot as plt

urls = ['http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
        'http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz']

def download_and_extract(data_dir, download_dir):
    for url in urls:
        target_file = url.split('/')[-1]
        if target_file not in os.listdir(download_dir):
            print('Downloading', url)
            urllib.request.urlretrieve(url, os.path.join(download_dir, target_file))
            tf = tarfile.open(url.split('/')[-1])
            tf.extractall(data_dir)
        else:
            print('Already downloaded', url)

def get_annotations(file_path):
    annotations = {}
    
    with open(file_path, 'r') as f:
        rows = f.read().splitlines()

    for i, row in enumerate(rows):
        image_name, _, _, _ = row.split(' ')
        class_name = image_name.split('_')[:-1]
        class_name = '_'.join(class_name)
        image_name = image_name + '.jpg'
        annotations[image_name] = class_name
    
    return annotations, i + 1


# In[5]:


if not os.path.isdir('data'):
    os.mkdir('data')

download_and_extract('data', '.')


# In[6]:


train_annotations, _ = get_annotations('data/annotations/trainval.txt')
test_annotations, _ = get_annotations('data/annotations/test.txt')


# In[7]:


all_annotations = {}

for key, value in train_annotations.items():
    all_annotations[key] = value
for key, value in test_annotations.items():
    all_annotations[key] = value


# In[8]:


train_annotations = {}
test_annotations = {}

for key, value in all_annotations.items():
    if random.randint(0, 99) < 20:
        test_annotations[key] = value
    else:
        train_annotations[key] = value


# In[9]:


train_count = len(list(train_annotations.keys()))
test_count = len(list(test_annotations.keys()))

print(train_count)
print(test_count)


# # Visualize Data

# In[10]:


classes = list(all_annotations.values())

classes = list(set(classes))

print(classes)
print('\nNum of classes:', len(classes))


# In[11]:


plt.figure(figsize=(8, 8))

train_images = list(train_annotations.keys())

for i in range(0, 8):
    plt.subplot(2, 4, i + 1)
    image = train_images[random.randint(0, train_count - 1)]
    plt.imshow(plt.imread(os.path.join('data/images/', image)))
    plt.xlabel(train_annotations[image])
plt.show()


# # SageMaker Setup

# In[12]:


role = sagemaker.get_execution_role
bucket_name = 'petsdatadogs'
training_image = get_image_uri(boto3.Session().region_name,'image-classification', repo_version ='latest')
print(training_image)


# In[13]:


folders = ['train', 'train_lst', 'validation', 'validation_lst']

for folder in folders:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)


# # Preparing Data for SageMaker

# In[14]:


def prepare_data(annotations, key='train'):
    images = list(annotations.keys())
    f = open(os.path.join(key + '_lst', key + '.lst'), 'w')
    with tqdm(total=len(images)) as pbar:
        for i, image in enumerate(images):
            shutil.copy(os.path.join('data/images/', image), os.path.join(key, image))
            class_id = classes.index(annotations[image])
            f.write('{}\t{}\t{}\n'.format(i, class_id, image))
            pbar.update(1)
    f.close()


# In[15]:


prepare_data(train_annotations, 'train')


# In[16]:


prepare_data(test_annotations, 'validation')


# # Uploading Data to S3

# In[18]:


get_ipython().run_cell_magic('time', '', "sess = sagemaker.Session()\ns3_train_path = sess.upload_data(path = 'train',bucket = bucket_name, \n                                 key_prefix='train')\nprint('training images uploaded')\ns3_train_lst_path = sess.upload_data(path = 'train_lst',bucket = bucket_name, \n                                     key_prefix='train_lst')\nprint('training list images uploaded')\ns3_validation_path = sess.upload_data(path = 'validation',bucket = bucket_name, \n                                      key_prefix='validation')\nprint('validation uploaded')\ns3_validation_lst_path = sess.upload_data(path = 'validation_lst',bucket = bucket_name, \n                                      key_prefix='validation_lst')\nprint('validation list uploaded')")


# In[ ]:





# # SageMaker Estimator

# In[19]:


model = sagemaker.estimator.Estimator(
        training_image,
role=role,
train_instance_count = 1,
train_instance_type ='ml.p3.2xlarge',
train_volume_size = 100,
train_max_run=36000,
input_mode='File',
output_path='s3://petsdatadogs/output',
sagemaker_session =sess)


# # Hyperparameters

# In[20]:


model.set_hyperparameters(
    num_layers=18,
    use_pretrained_model=1,
    image_shape='3,224,224',
    num_classes =37,
    mini_batch_size = 32,
    resize =224,
    epochs =10,
    learning_rate = 0.001,
    num_training_samples = train_count,
    augmentation_type = 'crop_color_transform',
    
)


# # Data Channels

# In[21]:


train_data=sagemaker.session.s3_input(s3_train_path, distribution = 'FullyReplicated',
                          content_type='application/x-image',s3_data_type='S3Prefix')
validation_data=sagemaker.session.s3_input(s3_validation_path, distribution = 'FullyReplicated',
                          content_type='application/x-image',s3_data_type='S3Prefix')
train_lst_data=sagemaker.session.s3_input(s3_train_lst_path, distribution = 'FullyReplicated',
                          content_type='application/x-image',s3_data_type='S3Prefix')
validation_lst_data=sagemaker.session.s3_input(s3_validation_lst_path, distribution = 'FullyReplicated',
                          content_type='application/x-image',s3_data_type='S3Prefix')


# In[22]:


data_channels={
    'train': train_data,
    'train_lst':train_lst_data,
    'validation':validation_data,
    'validation_lst':validation_lst_data
}


# # Model Training

# In[23]:


model.fit(inputs= data_channels,logs = True
         )


# # Deploy Model

# In[ ]:





# # Predictions

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Don't forget!! You need to delete endpoint or else you will continue to accrue cost!

# In[ ]:


sagemaker.Session().delete_endpoint(deployed_model.endpoint)


# In[ ]:




