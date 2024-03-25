import os.path
import shutil

import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets
from .serializers import ComtnfileSerializer,ComtnfiledetailSerializer
from .models import Comtnfile, Comtnfiledetail
from django.http import HttpResponse
from django.utils import timezone
from datetime import datetime
from django.conf import settings
from django.http import JsonResponse
from ultralytics import YOLO
from django.http import FileResponse
import matplotlib.pyplot as plt
import glob # predict에 저장된 이미지 전체를 가져와 이미지를 합치기위한 라이브러리임, 합친다기보단 경로상에 있는 파일을 다 긁어오기 위함임
import re # image_files(객체인식후 분할 저당 된 파일들 sort적용하기우한 라이브러리)
import copy

def conoly_api(request):
    return HttpResponse("sssssssssssssdsssss")




@csrf_exempt
def upload_file(request):
    if request.method == "POST":

        upload = request.FILES.getlist("atch_file")

        atch_file_id = str(Comtnfile.objects.count() + 1)
        index=0
        upload_yn=False

        if(len(upload) > 0):

            model_dir = os.path.join(settings.BASE_DIR, 'colony_api')
            model_path = os.path.join(model_dir, 'best.onnx')

            image_save_path = os.path.join(settings.BASE_DIR, 'upload')

            model = YOLO(model_path)
            file_details = []

            fileModel = Comtnfile(
                ATCH_FILE_ID=atch_file_id,
                CREAT_DT=timezone.now(),
                USE_AT='Y')

            fileModel.save()

            for f in upload :

                file_original_name, file_extsn = os.path.splitext(str(f))
                stre_file_name = str(f)[0] + "_" + datetime.now().strftime('%Y%m%d%H%M%S%f')

                # 이거 객체인식 이미지도 저장 하려면 필드 추가해야함
                # 객체인식 저장 경로만 있으면 됨 "PREDICT_STRE_COURS" 이런걸로
                fileDetailModel = Comtnfiledetail(
                    ATCH_FILE_ID = fileModel,
                    FILE_SN = index,
                    FILE_EXTSN = file_extsn,
                    STRE_FILE_NM = stre_file_name,
                    ORIGNL_FILE_NM =file_original_name,
                    FILE_STRE_COURS = settings.MEDIA_ROOT,
                    FILE_SIZE = f.size
                )
                index += 1  # 파일 순번 증가
                fileDetailModel.save()

                with open(settings.MEDIA_ROOT + "/" + stre_file_name + file_extsn, 'wb') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)

                img = cv2.imread(os.path.join(os.path.join(settings.MEDIA_ROOT, stre_file_name + file_extsn)))

                #original_height, original_width,  = img.shape[:2]
                original_height, original_width, = img.shape[:2]

                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

                ###################################################
                tile_size = 300

                image_tiles = split_image(img, tile_size)

                # 이미지 분할 시각화 (확인용)
                #visualize_image_tiles(img, image_tiles)

                idx = 0
                colony_count = 0
                predict_path = ''
                colony_count = 0
                for tile in image_tiles:
                    results = model.predict(source=tile, save=True)

                    for result in results:
                        predict_file_path = os.path.join(settings.BASE_DIR, result.save_dir, result.path)
                        predict_path = os.path.join(settings.BASE_DIR, result.save_dir)
                        os.rename(predict_file_path,os.path.join(settings.BASE_DIR, result.save_dir, stre_file_name + str(idx)+ file_extsn))
                        file_predict_save = result.save_dir #result 값이랑 DB 저장 때문에 쓰긴 하는데 이거 수정해야 함

                        if len(result.boxes.conf) > 0 :
                            colony_count += len(result.boxes.conf)

                        # print("cls : ",result.boxes.cls)
                        # print("conf : ",result.boxes.conf)
                        # print("conf.length : ",len(result.boxes.conf))
                        # print("orig_shape : ",result.boxes.orig_shape)
                        # print("shape : ",result.boxes.shape)

                    idx+=1

                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", colony_count)
                # 분할 된 개체 인식 파일을 모두 가져옴
                image_files = glob.glob(predict_path + "/*.jpg")

                # 이미지 파일 리스트를 복사하여 새로운 배열 생성
                sorted_image_files = copy.deepcopy(image_files)

                # 숫자로 변환한 파일 이름을 기준으로 정렬
                sorted_image_files = sorted(image_files, key=numerical_sort)

                # 파일 이름을 숫자로 변환하여 정렬
                sorted_image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

                # 파일 경로를 사용하여 이미지 로드
                images = [cv2.imread(file) for file in sorted_image_files]

                # 분할된 이미지 복원
                merged_image = merge_tiles(images, (original_height, original_width))

                merge_saved_path = os.path.join(settings.BASE_DIR, 'merge', stre_file_name + file_extsn)
                cv2.imwrite(merge_saved_path, merged_image)

                # 이미지 시각화 - 확인용
                #plt.imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))
                #plt.axis('off')  # 축 제거
                #plt.show()

                # 파일 디테일 정보를 딕셔너리로 만들어 리스트에 추가
                file_detail_info = {
                    'file_id': fileDetailModel.id,
                    'file_name': stre_file_name,
                    'file_orignl_name': file_original_name,
                    'file_size': f.size,
                    'file_extsn': file_extsn,
                    'file_path': settings.MEDIA_URL + stre_file_name + "." + file_extsn,
                    'merge_file_path': 'http://192.168.0.16:8000/colony_api/get_merge_file/'+ stre_file_name + file_extsn,
                    'colony_count' : colony_count
                    #'image_merge_save' : merge_saved_path
                }
                file_details.append(file_detail_info)

                # results = model.predict(source=os.path.join(settings.MEDIA_ROOT, stre_file_name + file_extsn), save=True)

            response_data = {'success': True, 'message': '파일이 성공적으로 업로드되었습니다.', 'files': file_details}
        else:
            response_data = {'success': False, 'message': '업로드된 파일이 없습니다..'}

        return JsonResponse(response_data)


def numerical_sort(value):
    # 파일명에서 숫자 부분을 추출하여 정수로 변환
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else float('inf')  # 숫자가 없는 경우 무한대 값 반환

def split_image(image, tile_size):
    """
    주어진 이미지를 타일 크기로 분할하여 타일 리스트를 반환합니다.
    """
    height, width = image.shape[:2]
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
    return tiles


def merge_tiles(tiles, original_size):
    """
    분할된 이미지 타일을 합쳐서 원래 이미지로 복원합니다.
    """
    original_height, original_width = original_size

    # 원본 이미지 크기에 맞는 빈 캔버스 생성
    reconstructed_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # 병합을 위한 인덱스 변수 설정
    index = 0

    # 이미지의 각 행에 대해 반복
    for y in range(0, original_height, tiles[0].shape[0]):
        for x in range(0, original_width, tiles[0].shape[1]):
            # 타일의 크기를 원본 이미지와 일치하도록 조정
            tile_height = min(tiles[index].shape[0], original_height - y)
            tile_width = min(tiles[index].shape[1], original_width - x)

            # 캔버스에 타일을 배치
            reconstructed_image[y:y + tile_height, x:x + tile_width] = tiles[index][:tile_height, :tile_width]
            index += 1

    return reconstructed_image


def visualize_image(image):
    """
    주어진 이미지를 시각화하여 출력합니다.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# 이미지 시각화 함수
def visualize_image_tiles(image, image_tiles):
    fig, ax = plt.subplots(1, len(image_tiles), figsize=(12, 6))
    if len(image_tiles) == 1:
        ax = [ax]
    for i, (tile, ax) in enumerate(zip(image_tiles, ax)):
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Tile {i}')
        ax.axis('off')
    plt.show()


# 이미지 리스트 잘 안나옴, 필요 있나?
def upload_file_list(request):
    if request.method == "GET":
        images = Comtnfiledetail.objects.all()
        return render(request, 'upload_file_list.html', {'images': images})

# 원본 이미지 불러오기
def get_origin_file(request, upload_file_name):
    if request.method == "GET":
        image = Comtnfiledetail.objects.get(STRE_FILE_NM=upload_file_name)
        file_path = os.path.join(image.FILE_STRE_COURS, image.STRE_FILE_NM + image.FILE_EXTSN)

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:

                response = HttpResponse(file.read(), content_type='image/jpeg')
                response['Content-Disposition'] = 'inline'

                # 아래처럼 하면 다운
                # response = HttpResponse(file.read(), content_type='application/octet-stream')
                # response['Content-Disposition'] = f'inline; filename="{os.path.basename(file_path)}"'
                return response
        else:
            return HttpResponse("File not found", status=404)

# 객체 인식 이미지 불러오기 이거 모델에 넣고 확장자까지 수정해야함
def get_predict_file(request, predict_order, upload_file_name):
    if request.method == "GET":
        file_path = os.path.join(settings.BASE_DIR, settings.MODEL_SAVE_ROOT, predict_order, upload_file_name )

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:

                response = HttpResponse(file.read(), content_type='image/jpeg')
                response['Content-Disposition'] = 'inline'
                return response
        else:
            return HttpResponse("File not found", status=404)


@csrf_exempt
def get_merge_file(request, upload_file_name):
    if request.method == "GET":
        file_path = os.path.join(settings.BASE_DIR, 'merge', upload_file_name )

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                response = HttpResponse(file.read(), content_type='image/jpeg')
                response['Content-Disposition'] = 'inline'
                return response
        else:
            return HttpResponse("File not found...", status=404)

    # POST 요청에 대한 처리 추가
    elif request.method == "POST":
        # POST 요청에 대한 처리 추가
        pass
