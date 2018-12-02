import requests
import os
import tqdm
from PIL import Image
from bs4 import BeautifulSoup

sess = requests.Session()
proxies = {}
if 'http_proxy' in os.environ:
    sess.proxies['http_proxy'] = os.environ['http_proxy']
if 'https_proxy' in os.environ:
    sess.proxies['https_proxy'] = os.environ['https_proxy']


def _is_img_file(filepath: str) -> bool:
    """
    check whether filepath is a valid image file

    :param filepath: image path to be checked
    :return: whether is a Image
    """
    # noinspection PyBroadException
    try:
        Image.open(filepath)
    except Exception:
        return False
    else:
        return True


def _download_image_list(file_lists: list, save_dir, idx) -> int:
    """
    download helper

    :param file_lists:
    :param save_dir:
    :param idx:
    :return: how many file are downloaded
    """
    downloaded = 0
    for img_link in tqdm.tqdm(file_lists, desc="Downloading page{:d}".format(idx),
                              leave=False,
                              dynamic_ncols=True,
                              unit='imgs',
                              unit_scale=True):
        img = sess.get(img_link, proxies=proxies)
        filename = img_link[img_link.rfind('/') + 1:]
        filepath = os.path.join(save_dir, filename)
        # print("[{:05d}/{:05d}]Downloading {:15s}".format(downloaded_times, wanna_size, filename), end='')
        with open(filepath, mode='wb') as f:
            f.write(img.content)
        if _is_img_file(filepath):
            # print('\t [OK]')
            downloaded += 1
        else:
            os.remove(filepath)
            # print('\t [FAILED]')
            downloaded -= 1
    return downloaded


def _check_downloaded_size(path, items_per_page):
    l = len(list(os.scandir(path)))
    return l, l // items_per_page


def download_konachan(save_dir: str, tag='hatsune_miku', wanna_size=5000, is_continue=True):
    print("Begin downloading \"{:s}\" from http://konachan.com"
          " \n\t save [{:d}] images to {:s}".format(tag, wanna_size, save_dir))

    # prepare parameters
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_continue:
        downloaded_times, pages = _check_downloaded_size(save_dir, 21)
    else:
        downloaded_times = 0
        pages = 1
    total = tqdm.tqdm(total=wanna_size, desc="Total",
                      leave=False, dynamic_ncols=True,
                      unit='imgs', unit_scale=True)
    total.update(downloaded_times)

    # download every page
    while downloaded_times < wanna_size:
        base_url = 'http://konachan.com/post?page={:d}&tags={:s}+order%3Afav+rating%3Asafe'.format(pages, tag)
        res = sess.get(base_url, proxies=proxies)

        # find all directlinks
        soup = BeautifulSoup(res.content, features="html.parser")
        links = soup.findAll(name='a', attrs={'class': 'directlink'})
        links = [a.attrs['href'] for a in links]

        # begin downloading
        downloaded = _download_image_list(links, save_dir, pages)
        pages += 1
        downloaded_times += downloaded
        total.update(downloaded)


def download_donmai(save_dir: str, tag='hatsune_miku', wanna_size=5000, is_continue=True):
    print("Begin downloading \"{:s}\" from https://danbooru.donmai.us"
          " \n\t save [{:d}] images to {:s}".format(tag, wanna_size, save_dir))

    # prepare parameters
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_continue:
        downloaded_times, pages = _check_downloaded_size(save_dir, 20)
    else:
        downloaded_times = 0
        pages = 1
    total = tqdm.tqdm(total=wanna_size, desc="Total",
                      leave=False, dynamic_ncols=True,
                      unit='imgs', unit_scale=True)
    total.update(downloaded_times)

    # download every page
    while downloaded_times < wanna_size:
        base_url = 'https://danbooru.donmai.us/posts?page={:d}&tags={:s}'.format(pages, tag)
        res = sess.get(base_url, proxies=proxies)

        # find all directlinks
        soup = BeautifulSoup(res.content, features="html.parser")
        links = soup.findAll('article', attrs={'class': 'post-preview'})
        links = [a.attrs['data-file-url'] for a in links]

        # begin downloading
        downloaded = _download_image_list(links, save_dir, pages)
        pages += 1
        downloaded_times += downloaded
        total.update(downloaded)


if __name__ == "__main__":
    download_donmai('/Volumes/Repository/Data/GAN/rem', tag='rem_(re:zero)', wanna_size=4500)
    download_konachan('/Volumes/Repository/Data/GAN/miku', tag='hatsune_miku', wanna_size=10000)
