from bs4 import BeautifulSoup
import requests
import os
import urllib
"""This is for scrapping memes off memegenerator.net
I'm super lucky the captions and tags are very scrappable"""

total_templates = 250
total_captions = 2000

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

    for page_num in range(1,total_templates):
        print('beginning image and caption download')
        if page_num == 1:
            url = 'https://memegenerator.net/memes/popular/alltime'
        else:
            url = 'https://memegenerator.net/memes/popular/alltime/page/' + str(page_num)

        img_links, links = find_img_links(url)

        for link, img in enumerate(img_links):
            url = img['src']
            response = requests.get(url)
            curr_img = url.split('/')[-1]
            full_path = current_path + meme_path + curr_img
            ### Get the templates
            with open(full_path, 'wb') as outfile:
                outfile.write(response.content)
            print(full_path)
            ### Get the captions
            for caption in range(1,total_captions):
                if caption == 1:
                    url = 'https://memegenerator.net' + links[link]['href']
                else:
                    url = 'https://memegenerator.net' + links[link]['href'] + '/images/popular/alltime/page/' + str(caption)

                title = curr_img.split('.jpg')[0]
                img_caption_links, caption_links = find_img_links(url)
                full_caption_path = current_path + meme_path + 'meme_captions_' + title + '.txt'
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
