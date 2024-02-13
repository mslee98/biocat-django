import os.path

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
            file_details = []

            fileModel = Comtnfile(
                ATCH_FILE_ID=atch_file_id,
                CREAT_DT=timezone.now(),
                USE_AT='Y')

            fileModel.save()

            for f in upload :

                file_original_name, file_extsn = os.path.splitext(str(f))
                stre_file_name = str(f)[0] + "_" + datetime.now().strftime('%Y%m%d%H%M%S%f')

                fileDetailModel = Comtnfiledetail(
                    ATCH_FILE_ID = fileModel,
                    FILE_SN = index,
                    FILE_EXTSN = file_extsn,
                    STRE_FILE_NM = stre_file_name,
                    ORIGNL_FILE_NM =file_original_name,
                    FILE_STRE_COURS = settings.MEDIA_ROOT
                )
                index += 1  # 파일 순번 증가

                with open(settings.MEDIA_ROOT + "/" + stre_file_name, 'wb') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)

                # 파일 디테일 정보를 딕셔너리로 만들어 리스트에 추가
                file_detail_info = {
                    'file_id': fileDetailModel.id,
                    'file_name': stre_file_name,
                    'file_orignl_name' : file_original_name,
                    'file_size': f.size,
                    'file_extsn': file_extsn,
                    'file_path': settings.MEDIA_URL + stre_file_name
                }
                file_details.append(file_detail_info)

            response_data = {'success': True, 'message': '파일이 성공적으로 업로드되었습니다.', 'files': file_details}
        else:
            response_data = {'success': False, 'message': '업로드된 파일이 없습니다..'}

        return JsonResponse(response_data)

