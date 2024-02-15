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

                fileDetailModel = Comtnfiledetail(
                    ATCH_FILE_ID = fileModel,
                    FILE_SN = index,
                    FILE_EXTSN = file_extsn,
                    STRE_FILE_NM = stre_file_name,
                    ORIGNL_FILE_NM =file_original_name,
                    FILE_STRE_COURS = settings.MEDIA_ROOT
                )
                index += 1  # 파일 순번 증가

                with open(settings.MEDIA_ROOT + "/" + stre_file_name + file_extsn, 'wb') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)

                img = cv2.imread(os.path.join(os.path.join(settings.MEDIA_ROOT, stre_file_name + file_extsn)))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

                # img_gray스케일로 변환하면 axis 형식이 맞지 않아 다시 RGB형식으로 변환해서 제공
                results = model.predict(source=img_rgb, save=True)

                file_predict_save = ''

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



