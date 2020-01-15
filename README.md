# book-ml-lesson

# 実行時のメモ

## サンプルコードのエラーに関して

p.54のbmi.pyでエラーが以下のエラーが表示された場合：

```
UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
```

この場合、Python環境で日本語が無効になっています。以下のコマンドを実行して、日本語が有効になるようにしてください。

```
localedef -f UTF-8 -i ja_JP ja_JP
localectl set-locale LANG=ja_JP.UTF-8
```

