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

def conoly_api(request):
    return HttpResponse("ssssssssssssssssss")



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

                print(stre_file_name+"#################################################")

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
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

                # img_gray스케일로 변환하면 axis 형식이 맞지 않아 다시 RGB형식으로 변환해서 제공
                results = model.predict(source=img_rgb, save=True)

                file_predict_save = ''

                # 이미지 변환 후 예측모델을 실행하면 default로 Image0으로 저장되어 덮어쓰기 진행, 방지하기 위한 소스코드
                for result in results:
                    predict_file_path = os.path.join(settings.BASE_DIR,result.save_dir,result.path)
                    os.rename(predict_file_path, os.path.join(settings.BASE_DIR, result.save_dir, stre_file_name + file_extsn))
                    file_predict_save = result.save_dir

                # 파일 디테일 정보를 딕셔너리로 만들어 리스트에 추가
                file_detail_info = {
                    'file_id': fileDetailModel.id,
                    'file_name': stre_file_name,
                    'file_orignl_name': file_original_name,
                    'file_size': f.size,
                    'file_extsn': file_extsn,
                    'file_path': settings.MEDIA_URL + stre_file_name,
                    'file_predict_save' : file_predict_save
                }
                file_details.append(file_detail_info)

                # results = model.predict(source=os.path.join(settings.MEDIA_ROOT, stre_file_name + file_extsn), save=True)

            response_data = {'success': True, 'message': '파일이 성공적으로 업로드되었습니다.', 'files': file_details}
        else:
            response_data = {'success': False, 'message': '업로드된 파일이 없습니다..'}

        return JsonResponse(response_data)

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

