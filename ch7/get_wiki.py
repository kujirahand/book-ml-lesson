# -*- coding: utf-8 -*-
import sys
import codecs
import re
import urllib.parse
import urllib.request
from janome.tokenizer import Tokenizer
from html.parser import HTMLParser

# ダウンロードする記事
urls = [ \
	('共和政ローマ', '1.txt'), \
	('王政ローマ', '2.txt'), \
	('不思議の国のアリス', '3.txt'), \
	('ふしぎの国のアリス', '4.txt'), \
	('Python', '5.txt'), \
	('Ruby', '6.txt'), \
]

# HTMLパーサー
class MyParser(HTMLParser):
	def __init__(self, **args):
		self.inptag = False
		self.tagdata = []
		self.current = ''
		super(MyParser, self).__init__(**args)
		
	def handle_starttag(self, tag, attrs):
		if tag == 'p':
			self.inptag = True

	def handle_endtag(self, tag):
		if tag == 'p':
			self.inptag = False
			self.tagdata.append(self.current)
			self.current = ''

	def handle_data(self, data):
		if self.inptag:
			self.current = self.current + data

# 形態素解析
tk = Tokenizer()

for url, dst in urls:
	# 日本語版Wikipediaのページ
	with urllib.request.urlopen('https://ja.wikipedia.org/wiki/'+ \
			urllib.parse.quote_plus(url)) as response:
		# URLから読み込む
		html = response.read().decode('utf-8')
	
		# 本文の<p>タグを取得する
		p = MyParser()
		p.feed(html)
	
		# 本文のみを取り出す
		with open(dst, 'w') as file:
			for a in p.tagdata:
				# 単語のリストにする
				l = [p.surface for p in tk.tokenize(a)]
				l = list(filter(lambda a: a.strip() != '', l))
				# 5単語以上
				if len(l) > 5:
					line = ' '.join(l)
					file.write(line)
					file.write('\n')
	
