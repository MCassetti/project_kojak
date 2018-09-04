from bs4 import BeautifulSoup
import requests
import os
import urllib
"""This is for scrapping memes off memegenerator.net
I'm super lucky the captions and tags are very scrappable"""

total_templates = 250
total_captions = 2000
start_template = 1
start_caption = 1
start_link = 0

def find_img_links(url):
    response = requests.get(url)
    if response:
        soup = BeautifulSoup(response.text,'html.parser')
        imgs = soup.find_all(class_ ='char-img')
        links = [img.find('a') for img in imgs]
        img_links = [img.find('img') for img in imgs]
    else:
        img_links = []
        links = []
    return img_links, links

if __name__ == "__main__":

    current_path = os.getcwd()
    meme_path = '/meme_downloads/'
    base_path = current_path + meme_path
    restart_file = base_path + 'caption_template.txt'

    for page_num in range(start_template,total_templates):

        if page_num == 1:
            url = 'https://memegenerator.net/memes/popular/alltime'
        else:
            url = 'https://memegenerator.net/memes/popular/alltime/page/' + str(page_num)

        img_links, links = find_img_links(url)

        for link, img in enumerate(img_links):
            url = img['src']
            response = requests.get(url)
            curr_img = url.split('/')[-1]
            title = curr_img.split('.jpg')[0]
            full_caption_path = base_path + 'meme_captions_' + title + '.txt'

            full_path = base_path + curr_img
            print(full_path)
            if os.path.isfile(full_caption_path): #don't need to visit this meme
                continue

            ### Get the templates
            with open(full_path, 'wb') as outfile:
                outfile.write(response.content)

            ### Get the captions

            for caption in range(start_caption,total_captions):

                ## restart logging
                with open(restart_file,'w') as fp:
                    fp.write('%s\t%s\t%s' % (page_num,caption,link + start_link))

                if caption == 1:
                    url = 'https://memegenerator.net' + links[link]['href']
                else:
                    url = 'https://memegenerator.net' + links[link]['href'] + '/images/popular/alltime/page/' + str(caption)


                img_caption_links, caption_links = find_img_links(url)

                if caption % 100 == 0:
                    print(caption) # logging purposes


                if caption_links:
                    with open(full_caption_path, 'a') as f:
                        for caption in caption_links:
                            tag1 = caption.findAll("div", class_="optimized-instance-text0")
                            tag2 = caption.findAll("div", class_="optimized-instance-text1")
                            top_caption_txt = str(tag1).replace('<','>').split('>')[6]
                            bottom_caption_text = str(tag2).replace('<','>').split('>')[6]
                            tag_txt = curr_img.replace('-',' ').split('.jpg')[0]
                            caption_txt = caption['href'].replace('-',' ').split('/')[-1]
                            f.write('tag: "%s"\ttop caption: "%s"\t bottom_caption: "%s"\n' % (tag_txt, top_caption_txt, bottom_caption_text))
                else:
                    break #we are done with this meme
