import os



current_dir = os.getcwd()
data_dir = '/meme_downloads/'
meme_caption_path = current_dir + data_dir
meme_text = os.listdir(meme_caption_path)
pause_token = ' <pause> ' #hoping to strip out later anyways
img_id = 0 
caption_id = 0
for meme in meme_text:
    img_id +=1 # image ids
    with open(meme_caption_path + meme,'rb') as f:
        for line in f:
            curr_caption = []
            current_line = line.decode(errors='ignore').strip('\n').split('\t')
            tag = current_line[0].replace('top caption:','').replace('\"','')
            top_caption = current_line[1].replace('top caption:','').replace('\"','')
            bottom_caption = current_line[2].replace('bottom_caption:','').replace('\"','')
            caption = top_caption.lstrip() + pause_token + bottom_caption.lstrip()

            images = tags + '.jpg'
            caption_id += 1
            print(caption,tag)
            break
        break
