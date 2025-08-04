import os
import subprocess
import sys
import zipfile
import requests
from tqdm import tqdm

def run_command(command):
    """지정된 명령을 실행하고 오류가 발생하면 스크립트를 종료합니다."""
    print(f"실행 중: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e.stderr.decode('utf-8')}")
        sys.exit(1)

def download_file(url, filename):
    """지정된 URL에서 파일을 다운로드하고 진행률을 표시합니다."""
    print(f"{url} 에서 {filename} 다운로드 중...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(filename, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"다운로드 오류 발생: {e}")
        sys.exit(1)


def main():
    """vast.ai 환경 설정을 위한 메인 스크립트"""
    print("Vast.ai 환경 설정 스크립트를 시작합니다.")

    # 1. pip 업그레이드
    run_command(["python", "-m", "pip", "install", "--upgrade", "pip==24.0"])

    # 1.5 requirements.txt 설치
    run_command(["pip", "install", "-r", "requirements.txt"])

    # 2. 필수 파일 다운로드 및 압축 해제
    prerequisites_url = "https://huggingface.co/Essid/ReFlow-VAE-SVC-Prerequisites/resolve/main/Prerequisites.zip"
    zip_filename = "Prerequisites.zip"
    pretrain_dir = "pretrain"

    download_file(prerequisites_url, zip_filename)

    print(f"'{pretrain_dir}' 폴더에 압축 해제 중...")
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(pretrain_dir)
        print("압축 해제가 완료되었습니다.")
    except zipfile.BadZipFile:
        print("오류: 잘못된 ZIP 파일입니다.")
        sys.exit(1)
    finally:
        # 다운로드한 zip 파일 삭제
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            print(f"'{zip_filename}' 파일을 삭제했습니다.")


    # 3. 파일 존재 여부 검사
    print("'pretrain' 폴더의 내용물을 확인합니다.")
    required_dirs = ["rmvpe", "contentvec", "nsf_hifigan"]
    missing_dirs = []

    for dirname in required_dirs:
        if not os.path.isdir(os.path.join(pretrain_dir, dirname)):
            missing_dirs.append(dirname)

    if missing_dirs:
        print("\n오류: 다음 필수 폴더가 'pretrain' 디렉터리에 없습니다:")
        for dirname in missing_dirs:
            print(f"- {dirname}")
        sys.exit(1)
    else:
        print("필수 폴더가 모두 존재합니다: " + ", ".join(required_dirs))


    # 4. 종료 메시지
    print("\n========================================")
    print("✅ 모든 설정 작업이 성공적으로 완료되었습니다.")
    print("========================================")

if __name__ == "__main__":
    main()
