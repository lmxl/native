__author__ = 'ymo'

import sys, json

sys.path.append("../stone/")
from os import listdir
from getFiles import fetch_file
from readability.readability import Document
from bs4 import BeautifulSoup
from multiprocessing import Pool
import pandas as pd
import HTMLParser, traceback
from tidylib import tidy_document
from time import gmtime, strftime
from os.path import exists
from re import sub
from itertools import product
labels ={}
disableQ = True
def cleanText(str):
    #str = sub('[?!\'\-,\.\|():"\]',' ', str)
    str = sub('[_\W]',' ', str)
    str = sub(' [^ ]{16}', ' ', str)
    str = sub(' +..? +', ' ', str)
    str = sub(' +..? +', ' ', str)
    str = sub(' +..? +', ' ', str)
    str = sub('  +', ' ', str)
    return str
def cleanStr(str):
    str = sub('[-\.?]','', str)
    str = sub('[0-9]','0', str)
    return str
def quadratic(pref, ls):
    rs = []
    if disableQ:
        return rs
    for i in range(len(ls)):
        for j in range(i+1, len(ls)):
            left = ls[i]
            right =ls[j]
            rs.append(pref + left + 'X' + right)
    return list(set(rs))
def tagFeatures(summary_html):
    tdoc = set([])
    for tc in ['p', 'li', 'div', 'ul']:
        for tag in summary_html.find_all(tc):
            classes = []
            attrs = tag.attrs
            if 'class' in attrs:
                for cls in attrs['class']:
                    classes.append(cleanStr(cls))
            for key in attrs:
                if key not in ['class', 'id', 'src', 'alt']:
                    classes.append(cleanStr(key))
            for cls in classes:
                tdoc.add(tc + cls)
    return list(tdoc)
def urlFeatures(summary_html):
    adoc = ['']
    urls = []
    for tag in summary_html.find_all('a'):
        #adoc.append('href')
        attrs = tag.attrs
        if 'href' in attrs:
            for cls in attrs['href'][7:].split('/')[:2]:
                if len(cls) > 30 or len(cls)<2:
                    continue
                urls.append(cleanStr(cls))
                adoc.append('aU' + cleanStr(cls))
            adoc.extend(quadratic('aUQ', urls))
    return list(set(adoc))
def imgFeatures(summary_html):
    add_text = [""]
    imgdoc = [""]
    for tag in summary_html.find_all('img'):
        #imgdoc.append('img')
        classes = []
        urls = []
        attrs = tag.attrs
        if 'class' in attrs:
            for cls in attrs['class']:
                classes.append(cleanStr(cls))
        for key in attrs:
            if key not in ['class', 'id', 'src', 'alt']:
                classes.append(cleanStr(key))
        for cls in classes:
            imgdoc.append('imgC' + cls)
        imgdoc.extend(quadratic('imgCQ', classes))
        if 'src' in attrs and attrs['src'].startswith('http://'):
            for cls in attrs['src'][7:].split('/')[:3]:
                imgdoc.append('imgU' + cleanStr(cls))
                urls.append(cleanStr(cls))
        imgdoc.extend(quadratic('imgUQ', urls))
        if 'alt' in attrs:
            add_text.append(attrs['alt'])
    return list(set(imgdoc)), add_text
def processFile(filepath):
    title = ""
    short = ""
    fine_text = ""
    filename = filepath.split('/')[-1]
    if filename in labels:
        label = labels[filename]
    else:
        label = -1
    try:
        #html_parser = HTMLParser.HTMLParser()
        raw = open(filepath).read().decode(encoding='ascii', errors='ignore')
        if raw.count(' ') * 3 > len(raw):
            raw = raw.replace('   ','')
        if False:
            tidy, errors = tidy_document(raw, options={'numeric-entities': 1})
            main_html = Document(tidy)
            summary = main_html.summary()
            summary_html = BeautifulSoup(summary, 'html.parser')
        else:
            summary_html = BeautifulSoup(raw, 'html.parser')
        imgdoc , add_text = imgFeatures(summary_html)
        adoc = urlFeatures(summary_html)
        tdoc = tagFeatures(summary_html)
        #print summary
        #print adoc, imgdoc, add_text
        coarse_text = summary_html.get_text()
        fine_text = ' '.join([i.strip() for i in coarse_text.split('\n') if len(i.strip()) > 2])
        fine_text = cleanText(fine_text + ' '.join(add_text))
        fine_text += ' '+' '.join(imgdoc) +' '+ ' '.join(adoc)+ ' ' +' '.join(tdoc)
        if summary_html.find('title') is not None and summary_html.find('title').string is not None:
            title = cleanText(summary_html.find('title').string)
        doc = {'label': label, 'title': title, 'short': short, 'text': fine_text, 'filename': filename}
        if fine_text.count(' ') * 3 > len(fine_text) or fine_text < 15:
            doc['failure'] = 2
            #fine_text = "None"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=4, file=sys.stdout)
        doc = {'label': label, 'title': title, 'short': short,
               'text': fine_text, 'filename': filename, 'failure': 1}
    return doc


home = fetch_file("https://onedrive.live.com/download?resid="
                  "CAE73F546D5A29CD!7657&authkey=!AC9BihmxjnqNX4E&ithint=file%2cjson")

def savejson(debug = False, unittest = False):
    global home
    if exists(home + 'summary.txt') and not unittest:
        print 'summary file already exists'
        return home
    files = []
    global labels
    labels = pd.read_csv(home + 'train.csv', index_col=0)['sponsored']
    for fd in ['0', '1', '2', '3', '4']:
        files.extend([home + fd + '/' + fn for fn in listdir(home + fd)])

    counter = 0
    fail = 0
    notify = 1
    pool = Pool(processes=13)  # start 4 worker processes
    it = pool.imap(processFile, files)
    if unittest:
        home = '/tmp/test'
        files= files[::1000]
    fp = open(home + 'summary.txt', 'w')
    for counter in range(1, len(files)+ 1):
        info = it.next()
        if 'failure' in info:
            fail += 1
        if counter == notify or counter % 10000 == 0:
            print '\r{time}\t{counter:7}({fail:6}) of {total}\t{name:20}\tLabel = {label:2}\tTitle = [ {title} ]...'.format(
                time=strftime("%Y-%m-%d %H:%M:%S", gmtime()), fail = fail,
                counter=counter, total=len(files), title=info['title'], name=info['filename'], label=info['label'])
            notify *= 2
        else:
            if 'failure' in info and unittest == True:
                print '\r{time}\t Failure in\t{name:20}\tLabel = {label:2}\tTitle = [ {title} ]...'.format(
                    time=strftime("%Y-%m-%d %H:%M:%S", gmtime()), counter=counter, total=len(files),
                    title=info['title'], name=info['filename'], label=info['label'])
        fp.write(json.dumps(info) + '\n')
    fp.close()
    return home

def runtest():
    global labels
    files = []
    counter = 0
    fail = 0
    labels = pd.read_csv(home + 'train.csv', index_col=0)['sponsored']
    for fd in ['0', '1', '2', '3', '4']:
        files.extend([home + fd + '/' + fn for fn in listdir(home + fd)])
    for file in files[::1000]:
        counter +=1
        info = processFile(file)
        if 'failure' in info:
            fail += 1
            print '\r{time}\t Failure in\t{name:25}\t{frac}\%\t{fail:4}/({counter:5})\tType = {type:2}\tContent = [ {content} ]'.format(
                time=strftime("%Y-%m-%d %H:%M:%S", gmtime()), counter=counter, fail = fail,frac = fail*100/counter,
                content=info['text'][:20], name=file, type=info['failure'])
if __name__ == "__main__":
    print processFile('/tmp/drive/native_data/0/1225146_raw_html.txt')
    runtest()


