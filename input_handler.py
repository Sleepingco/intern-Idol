# input_handler.py - 입력 처리 (키바인딩 등)
import pygame


def handle_keyboard_input(keys, config_vars):
    """키보드 입력: 필요한 키만 처리합니다.

    - ESC: 프로그램 종료
    """
    try:
        if keys[pygame.K_ESCAPE]:
            # 종료 플래그를 세팅하여 메인 루프에서 안전 종료
            config_vars["EXIT"] = True
    except Exception:
        # 환경에 따라 키 상태 조회 실패 시 무시
        pass


def update_display_caption(changed):
    """디스플레이 캡션 갱신 (키 안내 등)."""
    if changed:
        pygame.display.set_caption("Chin Strap AR")

