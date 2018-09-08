import os
import json


current_dir = os.getcwd()
data_dir = '/meme_downloads/'
meme_caption_path = current_dir + data_dir
meme_text = os.listdir(meme_caption_path)
pause_token = ' <pause> ' #hoping to strip out later anyways
image_id = 0
info_dict = {"description":"meme data set","keys":"image_id"}
id = 0
caption_dict = {}
image_dict = {}
full_caption_dict = {}
caption_list = []
image_list = []
for meme in meme_text:

    with open(meme_caption_path + meme,'rb') as f:
        for line in f:
            try:

                current_line = line.decode(errors='ignore').strip('\n').split('\t')
                tag = current_line[0].replace('top caption:','').replace('\"','')
                top_caption = current_line[1].replace('top caption:','').replace('\"','')
                bottom_caption = current_line[2].replace('bottom_caption:','').replace('\"','')
                caption = top_caption.lstrip() + pause_token + bottom_caption.lstrip()

                image = tag + '.jpg'
                caption_dict['image_id'] = image_id
                caption_dict['caption'] = caption
                caption_dict['id'] = id
                image_dict['file_name'] = image
                image_dict['image_id'] = image_id

                caption_list.append(caption_dict)
                image_list.append(image_dict)
                id += 1
                if id % 100 == 0:
                    print(caption,tag,image)
            except:
                "skipping caption, split did not work..."
                continue

        image_id +=1 # image ids

full_caption_dict['info'] = info_dict
full_caption_dict['captions'] = caption_list
full_caption_dict['images'] = image_list



with open('captions.json', 'w') as fp:
    json.dump(full_caption_dict, fp)
