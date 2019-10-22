# ETL1Cのファイルを読み込む
import struct
from PIL import Image, ImageEnhance, ImageOps
import glob, os

# ETL1ディレクトリ以下のファイルを処理する
def get_image():
    files = glob.glob("ETL1/*")
    for fname in files:
        # 情報ファイルは飛ばす
        if fname == "ETL1/ETL1INFO": continue
        if fname == "ETL1/README.md": continue
        print("read - " + fname)
        # ETL1のデータファイルを開く
        f = open(fname, 'rb')
        f.seek(0)
        while True:
            # 1レコード(メタデータと画像の2502バイト)を読む
            s = f.read(2052)
            if not s: break
            # バイナリデータをPythonが理解できるように抽出
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
            code_jis = r[3]
            image_data = r[18]
            # 必要なカナデータのみ抽出
            if not check_range(code_jis): continue
            # 画像データとして取り出す
            iF = Image.frombytes('F',
                (64, 63), image_data, 'bit', 4)
            iP = iF.convert('L')
            # 画像を鮮明にする
            enhancer = ImageEnhance.Brightness(iP)
            im_ehc = enhancer.enhance(16)
            # 画像と文字コードを戻す
            yield im_ehc, code_jis

def check_range(a):
    if a == 0 or a == 166 or (176 <= a <= 223):
        return True
    return False


if __name__ == '__main__':
    fmin, fmax = (255, 0)
    cnt = 0
    for i, o in enumerate(get_image()):
        img, code = o
        if code > fmax: fmax = code
        if (code > 0) and (code < fmin): fmin = code
        cnt = i
    print("range=", fmin, fmax)
    print("count=", cnt)

    
