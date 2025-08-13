import os, sys, json, copy, math, pygame, tkinter as tk
from tkinter import filedialog
from typing import List, Dict, Tuple, Any, Optional, cast

# ---------- CONFIG ----------
ASSET_DIR     = r"C:\Users\UserNameHere\Desktop\FolderName\SubFolder\SubFolder2"
FILENAME_BASE = "asset"
EXT           = ".png"
WORLD_W, WORLD_H = 864, 640

# UI sizes
TOPBAR_H    = 36
STATUS_H    = 22
LEFTBAR_W   = 72
RIGHTBAR_W  = 256
WIN_W       = LEFTBAR_W + 1024 + RIGHTBAR_W
WIN_H       = TOPBAR_H + 720 + STATUS_H

# Topbar controls (static rects)
SAVE_BTN_RECT = pygame.Rect(10, 6, 72, TOPBAR_H-12)

# ---------- XZEN THEME ----------
C_BG           = (10, 10, 12)
C_PANEL        = (18, 16, 24)
C_PANEL_DARK   = (12, 10, 18)
C_NEON         = (185, 100, 255)
C_NEON_DARK    = (120, 60, 200)
C_TEXT         = (235, 235, 245)
C_TEXT_DIM     = (175, 170, 195)
C_OK           = (120, 245, 170)
C_WARN         = (255, 110, 110)
C_EYE_OFF      = (90, 70, 120)
C_TOOLTIP_BG   = (16, 14, 22)

MASK_DRAW_COLOR     = (255, 255, 255)
MASK_ERASE_COLOR    = (0, 0, 0)
LINE_WIDTH_DEFAULT  = 8
DOUBLE_CLICK_MS     = 400

# Spawn overlays
SPAWN_COLOR   = (255, 64, 64)
SPAWN_BORDER  = (20, 10, 14)
SPAWN_SIZE    = 16

# Entry spawn overlay & bake colors
ENTRY_NEXT_OVERLAY = (255, 240, 120)
ENTRY_BACK_OVERLAY = (255, 120, 255)
ENTRY_MARK_SIZE    = 12
ENTRY_NEXT_BAKE_COLOR = (255, 255, 0)   # yellow
ENTRY_BACK_BAKE_COLOR = (255, 0, 255)   # magenta

# Door overlays + bake colors
DOOR_NEXT_OVERLAY_OUTLINE = (120, 255, 200)
DOOR_NEXT_OVERLAY_NODE    = (200, 255, 230)
DOOR_BACK_OVERLAY_OUTLINE = (120, 200, 255)
DOOR_BACK_OVERLAY_NODE    = (200, 230, 255)

DOOR_NEXT_BAKE_COLOR      = (0, 255, 0)   # forward
DOOR_BACK_BAKE_COLOR      = (0, 0, 255)   # back

pygame.init()
pygame.display.set_caption("Xzen Wall Engine 2D")
# double buffering helps a bit on Windows
screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.DOUBLEBUF)
clock  = pygame.time.Clock()
font       = pygame.font.SysFont("Tahoma,Segoe UI,Arial", 14)
font_small = pygame.font.SysFont("Tahoma,Segoe UI,Arial", 12)
font_tiny  = pygame.font.SysFont("Tahoma,Segoe UI,Arial", 10, bold=True)
font_mono  = pygame.font.SysFont("Consolas,Menlo,monospace", 12)

# ---------- UTILS ----------
def next_asset_path():
    os.makedirs(ASSET_DIR, exist_ok=True)
    i = 1
    while True:
        name = f"{FILENAME_BASE}{i}{EXT}"
        path = os.path.join(ASSET_DIR, name)
        if not os.path.exists(path): return path
        i += 1

def bevel_rect(surf, r, base, lt, dk, fill=True, radius=6):
    if fill: pygame.draw.rect(surf, base, r, border_radius=radius)
    pygame.draw.rect(surf, C_NEON_DARK, r, width=1, border_radius=radius)
    pygame.draw.line(surf, lt, (r.left+1, r.top+1), (r.right-2, r.top+1))
    pygame.draw.line(surf, lt, (r.left+1, r.top+1), (r.left+1, r.bottom-2))
    pygame.draw.line(surf, dk, (r.left+1, r.bottom-2), (r.right-2, r.bottom-2))
    pygame.draw.line(surf, dk, (r.right-2, r.top+1), (r.right-2, r.bottom-2))

def text(surf, s, pos, col=C_TEXT, f=font):
    surf.blit(f.render(s, True, col), pos)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def rdp_simplify(pts: List[Tuple[int,int]], eps: float):
    if len(pts) < 3: return pts[:]
    def dist(p, a, b):
        if a == b: return math.hypot(p[0]-a[0], p[1]-a[1])
        t = max(0, min(1, ((p[0]-a[0])*(b[0]-a[0]) + (p[1]-a[1])*(b[1]-a[1])) /
                         ((b[0]-a[0])**2 + (b[1]-a[1])**2)))
        proj = (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
        return math.hypot(p[0]-proj[0], p[1]-proj[1])
    def rec(s, e):
        max_d = 0; idx = -1
        for i in range(s+1, e):
            d = dist(pts[i], pts[s], pts[e])
            if d > max_d: max_d = d; idx = i
        if max_d > eps:
            left  = rec(s, idx)
            right = rec(idx, e)
            return left[:-1] + right
        else:
            return [pts[s], pts[e]]
    return rec(0, len(pts)-1)

# ---------- FILE PICKER ----------
root = tk.Tk(); root.withdraw()
BG_PATH = filedialog.askopenfilename(title="Select background PNG", filetypes=[("PNG","*.png")])
if not BG_PATH:
    print("❌ No background selected."); pygame.quit(); sys.exit()

bg_world = pygame.image.load(BG_PATH).convert()
bg_world = pygame.transform.scale(bg_world, (WORLD_W, WORLD_H)).convert()
mask_world = pygame.Surface((WORLD_W, WORLD_H)).convert()
mask_world.fill((0,0,0))

# Lazy edges: compute only on first toggle for faster startup
edges_overlay: Optional[pygame.Surface] = None
show_edges = False

def compute_edges_surface(bg):
    import numpy as np  # lazy import (pygame.surfarray requires numpy)
    arr = cast(Any, pygame.surfarray.pixels3d(bg)).copy()
    w,h = bg.get_width(), bg.get_height()
    edges = pygame.Surface((w,h), flags=0, depth=32).convert_alpha()
    px = cast(Any, pygame.surfarray.pixels_alpha(edges)); px[:] = 0  # type: ignore[index]
    # simple gradient magnitude
    for y in range(1,h-1):
        for x in range(1,w-1):
            gx = abs(int(arr[x+1,y].mean()) - int(arr[x-1,y].mean()))
            gy = abs(int(arr[x,y+1].mean()) - int(arr[x,y-1].mean()))
            g  = clamp(gx+gy,0,255)
            px[x,y] = g
    del px
    cast(Any, pygame.surfarray.pixels3d(edges))[:,:,:] = (C_NEON[0], C_NEON[1], C_NEON[2])  # type: ignore[index]
    edges.set_alpha(96)
    return edges

# ---------- DATA ----------
strokes: List[Dict[str, Any]] = []
points: List[Tuple[int,int]] = []
bezier_temp: List[Tuple[int,int]] = []

# doors: list of {pts:[(x,y)], visible, locked, name, kind:'next'|'back'}
doors: List[Dict[str, Any]] = []
door_points: List[Tuple[int,int]] = []

# entry spawn pixels (yellow ► / magenta ◄)
entry_next_spawns: List[Tuple[int,int]] = []
entry_back_spawns: List[Tuple[int,int]] = []

sel_kind: Optional[str] = None   # 'stroke' | 'door' | None
sel_idx: Optional[int]  = None

undo_stack: List[Any] = []
redo_stack: List[Any] = []

VIEW = pygame.Rect(LEFTBAR_W, TOPBAR_H, WIN_W-LEFTBAR_W-RIGHTBAR_W, WIN_H-TOPBAR_H-STATUS_H)
zoom = 1.0
ox, oy = (VIEW.x + (VIEW.w - WORLD_W)//2), (VIEW.y + (VIEW.h - WORLD_H)//2)
last_click_time = 0

TOOL_BRUSH="brush"; TOOL_LINE="line"; TOOL_MOVE="move"; TOOL_HAND="hand"; TOOL_BEZ="bezier"; TOOL_SPAWN="spawn"
TOOL_DOOR_NEXT="door_next"; TOOL_DOOR_BACK="door_back"
TOOL_ENTRY_NEXT="entry_spawn_next"; TOOL_ENTRY_BACK="entry_spawn_back"
tool = TOOL_BRUSH
create_tool = TOOL_BRUSH
mode = "create"

dragging = False
drag_started = False     # for one-undo-per-drag
last_mouse = (0,0)
space_pan = False
mmb_pan   = False

brush_w = LINE_WIDTH_DEFAULT
grid_on = False; grid_size = 16
preview_alpha = 96
simplify_on = False
sym_x = False; sym_y = False

renaming = False; rename_buf = ""

history_show = False
help_show    = False

spawn_pos: Optional[Tuple[int,int]] = None

# --- render scaling cache (perf) ---
_last_zoom = -1.0
_cached_bg: Optional[pygame.Surface] = None
_cached_mask: Optional[pygame.Surface] = None
_cached_edges: Optional[pygame.Surface] = None
_scaled_mask_dirty = True  # when mask_world changes, rescale once

# ---------- COORD HELPERS ----------
def world_to_screen(x: float, y: float) -> Tuple[int,int]:
    return (int(VIEW.x + ox + x*zoom), int(VIEW.y + oy + y*zoom))

def screen_to_world(sx: float, sy: float) -> Tuple[float,float]:
    return ((sx - VIEW.x - ox)/zoom, (sy - VIEW.y - oy)/zoom)

# ---------- CORE OPS ----------
def update_mask():
    global _scaled_mask_dirty
    mask_world.fill(MASK_ERASE_COLOR)
    for st in strokes:
        if not st.get('visible', True): continue
        draw_stroke_on(mask_world, st, MASK_DRAW_COLOR)
    _scaled_mask_dirty = True  # tell renderer to refresh scaled mask

def draw_stroke_on(surf, st, col):
    pts = st['pts']
    if st['mode'] == 'poly':
        for a,b in zip(pts, pts[1:]):
            pygame.draw.line(surf, col, a, b, brush_w)
    elif st['mode'] == 'straight_poly':
        for (x1,y1),(x2,y2) in zip(pts, pts[1:]):
            dx,dy = x2-x1, y2-y1
            if abs(dx) >= abs(dy):
                rect = pygame.Rect(min(x1,x2), y1-brush_w//2, abs(dx), brush_w)
            else:
                rect = pygame.Rect(x1-brush_w//2, min(y1,y2), brush_w, abs(dy))
            pygame.draw.rect(surf, col, rect)
    elif st['mode'] == 'bezier':
        if len(pts)>=3:
            a,c,b = pts[0], pts[1], pts[2]
            last = a
            steps = max(10, int(math.dist(a,b)/6))
            for i in range(1, steps+1):
                t = i/steps
                x = int((1-t)**2*a[0] + 2*(1-t)*t*c[0] + t**2*b[0])
                y = int((1-t)**2*a[1] + 2*(1-t)*t*c[1] + t**2*b[1])
                pygame.draw.line(surf, col, last, (x,y), brush_w)
                last = (x,y)

def stroke_bounds(st):
    xs=[p[0] for p in st['pts']]; ys=[p[1] for p in st['pts']]
    if not xs: return pygame.Rect(0,0,0,0)
    return pygame.Rect(min(xs)-brush_w, min(ys)-brush_w, max(xs)-min(xs)+2*brush_w, max(ys)-min(ys)+2*brush_w)

def door_bounds(d):
    xs=[p[0] for p in d['pts']]; ys=[p[1] for p in d['pts']]
    if not xs: return pygame.Rect(0,0,0,0)
    return pygame.Rect(min(xs)-4, min(ys)-4, max(xs)-min(xs)+8, max(ys)-min(ys)+8)

def hit_test_strokes(world_pos):
    for i in reversed(range(len(strokes))):
        if not strokes[i].get('visible', True): continue
        if stroke_bounds(strokes[i]).inflate(6,6).collidepoint(world_pos):
            return i
    return None

def hit_test_doors(world_pos):
    for i in reversed(range(len(doors))):
        if not doors[i].get('visible', True): continue
        if door_bounds(doors[i]).inflate(6,6).collidepoint(world_pos):
            return i
    return None

def commit_points(mode_kind, pts):
    if len(pts) < 2: return
    if simplify_on and mode_kind != 'bezier':
        pts = rdp_simplify(pts, eps=2.0)
    s = {'mode': mode_kind, 'pts': pts, 'visible': True, 'locked': False, 'name': f"Stroke {len(strokes):02d}"}
    strokes.append(s)
    update_mask()

def commit_door_points(pts, kind: str):
    if len(pts) < 3: return
    if simplify_on:
        pts = rdp_simplify(pts, eps=2.0)
    d = {'pts': [(int(x),int(y)) for (x,y) in pts], 'visible': True, 'locked': False,
         'name': f"Door {len(doors):02d}", 'kind': 'next' if kind!='back' else 'back'}
    doors.append(d)

def symmetry_mirror_pts(pts):
    out=[]
    if sym_x:
        cx = WORLD_W//2
        out += [(2*cx - x, y) for (x,y) in pts]
    if sym_y:
        cy = WORLD_H//2
        out += [(x, 2*cy - y) for (x,y) in pts]
    return out

# ---------- UNDO/REDO ----------
def snapshot_state():
    return copy.deepcopy((
        strokes, doors, sel_kind, sel_idx, brush_w, preview_alpha,
        grid_on, grid_size, simplify_on, sym_x, sym_y, spawn_pos, zoom, ox, oy,
        entry_next_spawns, entry_back_spawns
    ))

def restore_state(snap):
    global strokes, doors, sel_kind, sel_idx, brush_w, preview_alpha
    global grid_on, grid_size, simplify_on, sym_x, sym_y, spawn_pos, zoom, ox, oy
    global entry_next_spawns, entry_back_spawns, _scaled_mask_dirty, _last_zoom
    (strokes, doors, sel_kind, sel_idx, brush_w, preview_alpha,
     grid_on, grid_size, simplify_on, sym_x, sym_y, spawn_pos, zoom, ox, oy,
     entry_next_spawns, entry_back_spawns) = copy.deepcopy(snap)
    _scaled_mask_dirty = True
    _last_zoom = -1.0
    update_mask()

def push_undo():
    undo_stack.append(snapshot_state())
    if len(undo_stack) > 128: undo_stack.pop(0)
    redo_stack.clear()

def do_undo():
    if not undo_stack: return
    snap = undo_stack.pop()
    redo_stack.append(snapshot_state())
    restore_state(snap)

def do_redo():
    if not redo_stack: return
    snap = redo_stack.pop()
    undo_stack.append(snapshot_state())
    restore_state(snap)

# ---------- PANEL / UI ----------
def draw_save_row():
    bevel_rect(screen, SAVE_BTN_RECT, C_PANEL, C_NEON, C_PANEL_DARK)
    text(screen, "Save", (SAVE_BTN_RECT.x+18, SAVE_BTN_RECT.y+3), C_OK)
    return SAVE_BTN_RECT

def draw_status(msg_left):
    r = pygame.Rect(0, WIN_H-STATUS_H, WIN_W, STATUS_H)
    pygame.draw.rect(screen, C_PANEL, r)
    pygame.draw.line(screen, C_NEON_DARK, (0, WIN_H-STATUS_H), (WIN_W, WIN_H-STATUS_H))
    n_next = sum(1 for d in doors if d.get('kind','next')=='next')
    n_back = sum(1 for d in doors if d.get('kind','next')=='back')
    n_en = len(entry_next_spawns); n_eb = len(entry_back_spawns)
    right = f"Mode:{mode} Tool:{tool} W:{brush_w}px Zoom:{int(zoom*100)}% Grid:{'on' if grid_on else 'off'} SymX:{sym_x} SymY:{sym_y}"
    if spawn_pos: right += f"  Spawn:{int(spawn_pos[0])},{int(spawn_pos[1])}"
    right += f"  Doors►:{n_next} ◄:{n_back}  Entry►:{n_en} ◄:{n_eb}"
    text(screen, msg_left, (8, WIN_H-STATUS_H+3), C_TEXT_DIM, font_small)
    text(screen, right, (WIN_W-720, WIN_H-STATUS_H+3), C_TEXT_DIM, font_small)

def draw_tooltip(s, anchor_rect):
    pad = 6
    surf = font_small.render(s, True, C_TEXT)
    w,h = surf.get_width()+pad*2, surf.get_height()+pad*2
    x = min(anchor_rect.right + 10, WIN_W - w - 6)
    y = max(TOPBAR_H + 6, min(anchor_rect.y, WIN_H - STATUS_H - h - 6))
    rr = pygame.Rect(x, y, w, h)
    bevel_rect(screen, rr, C_TOOLTIP_BG, C_NEON_DARK, C_PANEL_DARK)
    screen.blit(surf, (rr.x+pad, rr.y+pad))

def draw_left_toolbar(mouse_pos):
    r = pygame.Rect(0, TOPBAR_H, LEFTBAR_W, WIN_H-TOPBAR_H-STATUS_H)
    bevel_rect(screen, r, C_PANEL_DARK, C_NEON_DARK, C_PANEL_DARK)

    btns = []
    size = 28
    gap  = 4
    inner_w = LEFTBAR_W - 12
    cols = max(1, (inner_w + gap) // (size + gap))  # auto-wrap
    start_x = 6
    start_y = r.y + 8

    items = [
        (TOOL_BRUSH, "B", "Brush (Create)"),
        (TOOL_LINE, "L", "Line (Create)"),
        (TOOL_BEZ, "Q", "Bezier (Create)"),
        (TOOL_DOOR_NEXT, "O", "Door► (Next)"),
        (TOOL_DOOR_BACK, "U", "Door◄ (Back)"),
        (TOOL_MOVE, "Mv", "Move (Edit)"),
        (TOOL_HAND, "H", "Hand pan"),
        (TOOL_SPAWN, "P", "Spawn marker"),
        (TOOL_ENTRY_NEXT, "Y", "Entry► (yellow)"),
        (TOOL_ENTRY_BACK, "M", "Entry◄ (magenta)"),
        ("dup", "Dup", "Duplicate selected"),
        ("del", "Del", "Delete selected"),
        ("clear", "Clr", "Clear all"),
        ("fit", "Fit", "Fit image to view"),
    ]

    hovered = None
    for idx, (key, label, tip) in enumerate(items):
        col_idx = idx % cols
        row_idx = idx // cols
        bx = start_x + col_idx * (size + gap)
        by = start_y + row_idx * (size + gap)
        b = pygame.Rect(bx, by, size, size)

        active = (key == tool) or (
            key in (TOOL_BRUSH, TOOL_LINE, TOOL_BEZ, TOOL_DOOR_NEXT, TOOL_DOOR_BACK)
            and key == create_tool and mode == "create"
        )
        col = C_NEON if active else C_PANEL
        bevel_rect(screen, b, col, C_NEON_DARK, C_PANEL_DARK)
        text(screen, label, (b.centerx - font_tiny.size(label)[0] // 2, b.centery - 6), C_TEXT, font_tiny)
        btns.append((key, b, tip))
        if b.collidepoint(mouse_pos): hovered = (tip, b)
    return btns, hovered

def draw_right_panel():
    r = pygame.Rect(WIN_W-RIGHTBAR_W, TOPBAR_H, RIGHTBAR_W, WIN_H-TOPBAR_H-STATUS_H)
    bevel_rect(screen, r, C_PANEL_DARK, C_NEON_DARK, C_PANEL_DARK)
    text(screen, "Layers  (F2 rename  ·  L lock  ·  Eye click)", (r.x+12, r.y+8), C_TEXT_DIM, font_small)
    list_r = pygame.Rect(r.x+10, r.y+28, r.w-20, r.h-38)
    pygame.draw.rect(screen, C_PANEL, list_r, border_radius=6)
    rows=[]; row_h=28; yoff=list_r.y

    # Walls
    text(screen, "Walls", (list_r.x+8, yoff-2), C_TEXT_DIM, font_small); yoff += 6
    for i,st in enumerate(strokes):
        y = yoff + i*row_h
        if y > list_r.bottom-row_h: break
        rr = pygame.Rect(list_r.x+4, y+2, list_r.w-8, row_h-6)
        active = (sel_kind=="stroke" and sel_idx==i)
        base = (30, 26, 42) if active else (22, 20, 32)
        bevel_rect(screen, rr, base, C_NEON_DARK, C_PANEL_DARK)
        eye = pygame.Rect(rr.x+6, rr.y+5, 18, 18)
        pygame.draw.rect(screen, C_PANEL_DARK if st.get('visible',True) else C_EYE_OFF, eye, border_radius=4)
        pygame.draw.circle(screen, C_TEXT if st.get('visible',True) else C_PANEL, eye.center, 6, 2)
        lock_r = pygame.Rect(eye.right+6, rr.y+6, 14, 14)
        pygame.draw.rect(screen, (60,60,80), lock_r, 1, border_radius=3)
        if st.get('locked',False):
            pygame.draw.line(screen, C_WARN, (lock_r.left+2, lock_r.top+2), (lock_r.right-2, lock_r.bottom-2), 2)
            pygame.draw.line(screen, C_WARN, (lock_r.left+2, lock_r.bottom-2), (lock_r.right-2, lock_r.top+2), 2)
        nm = st.get('name', f"Stroke {i:02d}")
        label_col = C_TEXT_DIM if not st.get('visible',True) else (C_WARN if st.get('locked',False) else C_TEXT)
        if renaming and sel_kind=="stroke" and sel_idx==i:
            pygame.draw.rect(screen, (50,50,70), pygame.Rect(lock_r.right+6, rr.y+4, 140, 18), 0, 3)
            text(screen, rename_buf, (lock_r.right+10, rr.y+5), C_TEXT, font_small)
        else:
            text(screen, nm, (lock_r.right+8, rr.y+5), label_col, font_small)
        rows.append(("stroke", i, rr, eye, lock_r))
    yoff += max(24, len(strokes)*row_h + 12)

    # Doors
    text(screen, "Doors", (list_r.x+8, yoff), C_TEXT_DIM, font_small); yoff += 6
    for i,d in enumerate(doors):
        y = yoff + i*row_h
        if y > list_r.bottom-row_h: break
        rr = pygame.Rect(list_r.x+4, y+2, list_r.w-8, row_h-6)
        active = (sel_kind=="door" and sel_idx==i)
        base = (26, 28, 40) if active else (18, 22, 30)
        bevel_rect(screen, rr, base, C_NEON_DARK, C_PANEL_DARK)
        eye = pygame.Rect(rr.x+6, rr.y+5, 18, 18)
        pygame.draw.rect(screen, C_PANEL_DARK if d.get('visible',True) else C_EYE_OFF, eye, border_radius=4)
        pygame.draw.circle(screen, C_TEXT if d.get('visible',True) else C_PANEL, eye.center, 6, 2)
        lock_r = pygame.Rect(eye.right+6, rr.y+6, 14, 14)
        pygame.draw.rect(screen, (60,60,80), lock_r, 1, border_radius=3)
        if d.get('locked',False):
            pygame.draw.line(screen, C_WARN, (lock_r.left+2, lock_r.top+2), (lock_r.right-2, lock_r.bottom-2), 2)
            pygame.draw.line(screen, C_WARN, (lock_r.left+2, lock_r.bottom-2), (lock_r.right-2, lock_r.top-2), 2)
        kind = d.get('kind','next')
        tag  = "►" if kind=='next' else "◄"
        nm = f"{tag} " + d.get('name', f"Door {i:02d}")
        label_col = (180,255,220) if kind=='next' else (180,220,255)
        if not d.get('visible',True): label_col = C_TEXT_DIM
        if d.get('locked',False):     label_col = C_WARN
        if renaming and sel_kind=="door" and sel_idx==i:
            pygame.draw.rect(screen, (50,50,70), pygame.Rect(lock_r.right+6, rr.y+4, 160, 18), 0, 3)
            text(screen, rename_buf, (lock_r.right+10, rr.y+5), C_TEXT, font_small)
        else:
            text(screen, nm, (lock_r.right+8, rr.y+5), label_col, font_small)
        rows.append(("door", i, rr, eye, lock_r))

    return rows

def draw_viewport():
    global _last_zoom, _cached_bg, _cached_mask, _cached_edges, _scaled_mask_dirty

    pygame.draw.rect(screen, C_PANEL, VIEW, 0, border_radius=6)
    clip_old = screen.get_clip(); screen.set_clip(VIEW)

    # checker bg
    tile=16
    for yy in range(VIEW.y, VIEW.bottom, tile):
        for xx in range(VIEW.x, VIEW.right, tile):
            c = C_PANEL_DARK if ((xx//tile + yy//tile) % 2 == 0) else C_PANEL
            pygame.draw.rect(screen, c, pygame.Rect(xx,yy,tile,tile))

    # recompute scaled caches only when needed
    if _last_zoom != zoom:
        target_size = (int(WORLD_W*zoom), int(WORLD_H*zoom))
        _cached_bg = pygame.transform.scale(bg_world, target_size)
        if edges_overlay is not None:
            _cached_edges = pygame.transform.scale(edges_overlay, target_size)
        _cached_mask = pygame.transform.scale(mask_world, target_size)
        _last_zoom = zoom
        _scaled_mask_dirty = False
    elif _scaled_mask_dirty:
        target_size = (int(WORLD_W*zoom), int(WORLD_H*zoom))
        _cached_mask = pygame.transform.scale(mask_world, target_size)
        _scaled_mask_dirty = False

    if _cached_bg is not None:
        screen.blit(_cached_bg, (VIEW.x+ox, VIEW.y+oy))
    if _cached_mask is not None:
        m = _cached_mask.copy(); m.set_alpha(preview_alpha); screen.blit(m, (VIEW.x+ox, VIEW.y+oy))
    if show_edges and _cached_edges is not None:
        screen.blit(_cached_edges, (VIEW.x+ox, VIEW.y+oy))

    # live previews (create mode)
    if mode=="create":
        col=(80,255,160)
        if create_tool in (TOOL_BRUSH, TOOL_LINE) and points:
            pts = points[:]
            mx,my = pygame.mouse.get_pos()
            wx,wy = screen_to_world(mx,my)
            if pygame.key.get_mods() & pygame.KMOD_SHIFT and pts:
                ax,ay=pts[-1]
                if abs(wx-ax) > abs(wy-ay): wy = ay
                else: wx = ax
            if grid_on:
                wx = round(wx / grid_size) * grid_size
                wy = round(wy / grid_size) * grid_size
            pts2 = pts + [(int(wx),int(wy))]
            for a, b in zip(pts2, pts2[1:]):
                ax,ay = world_to_screen(*a); bx,by = world_to_screen(*b)
                pygame.draw.line(screen, col, (ax,ay), (bx,by), max(1, int(brush_w*zoom)))
        if create_tool == TOOL_BEZ and len(bezier_temp)>0:
            for p in bezier_temp:
                sx,sy = world_to_screen(*p); pygame.draw.circle(screen, (160,220,255), (sx,sy), 4)
        if create_tool in (TOOL_DOOR_NEXT, TOOL_DOOR_BACK) and door_points:
            mx,my = pygame.mouse.get_pos()
            wx,wy = screen_to_world(mx,my)
            if grid_on:
                wx = round(wx / grid_size) * grid_size
                wy = round(wy / grid_size) * grid_size
            pts = door_points + [(int(wx),int(wy))]
            outline = DOOR_NEXT_OVERLAY_OUTLINE if create_tool==TOOL_DOOR_NEXT else DOOR_BACK_OVERLAY_OUTLINE
            node    = DOOR_NEXT_OVERLAY_NODE    if create_tool==TOOL_DOOR_NEXT else DOOR_BACK_OVERLAY_NODE
            for a,b in zip(pts, pts[1:]):
                ax,ay = world_to_screen(*a); bx,by = world_to_screen(*b)
                pygame.draw.line(screen, outline, (ax,ay), (bx,by), 2)
            for p in door_points:
                sx,sy = world_to_screen(*p); pygame.draw.circle(screen, node, (sx,sy), 4)

    # selection bounds
    if sel_kind=="stroke" and sel_idx is not None and 0<=sel_idx<len(strokes):
        b = stroke_bounds(strokes[sel_idx])
        tl=world_to_screen(b.left,b.top); br=world_to_screen(b.right,b.bottom)
        pygame.draw.rect(screen, C_NEON, pygame.Rect(tl,(br[0]-tl[0], br[1]-tl[1])), 2)
    if sel_kind=="door" and sel_idx is not None and 0<=sel_idx<len(doors):
        b = door_bounds(doors[sel_idx])
        tl=world_to_screen(b.left,b.top); br=world_to_screen(b.right,b.bottom)
        pygame.draw.rect(screen, (250,230,120), pygame.Rect(tl,(br[0]-tl[0], br[1]-tl[1])), 2)

    # draw doors overlay
    for d in doors:
        if not d.get('visible', True): continue
        pts = d['pts']; kind = d.get('kind','next')
        if len(pts)>=2:
            outline = DOOR_NEXT_OVERLAY_OUTLINE if kind=='next' else DOOR_BACK_OVERLAY_OUTLINE
            for a,b in zip(pts, pts[1:]):
                ax,ay = world_to_screen(*a); bx,by = world_to_screen(*b)
                pygame.draw.line(screen, outline, (ax,ay), (bx,by), 2)
            if len(pts)>=3:
                ax,ay = world_to_screen(*pts[-1]); bx,by = world_to_screen(*pts[0])
                pygame.draw.line(screen, outline, (ax,ay), (bx,by), 2)

    # grid
    if grid_on:
        gcol=(40,40,60)
        step=int(grid_size*zoom)
        offx=int((VIEW.x+ox)%step); offy=int((VIEW.y+oy)%step)
        for x in range(VIEW.x+offx, VIEW.right, step): pygame.draw.line(screen, gcol, (x, VIEW.y), (x, VIEW.bottom))
        for y in range(VIEW.y+offy, VIEW.bottom, step): pygame.draw.line(screen, gcol, (VIEW.x, y), (VIEW.right, y))

    # global spawn marker
    if spawn_pos is not None:
        sx, sy = world_to_screen(spawn_pos[0], spawn_pos[1])
        sz = max(2, int(SPAWN_SIZE * zoom))
        rect = pygame.Rect(sx - sz//2, sy - sz//2, sz, sz)
        pygame.draw.rect(screen, SPAWN_COLOR, rect)
        pygame.draw.rect(screen, SPAWN_BORDER, rect, 2)

    # entry spawn markers
    for x,y in entry_next_spawns:
        sx, sy = world_to_screen(x,y)
        sz = max(2, int(ENTRY_MARK_SIZE * zoom))
        r = pygame.Rect(sx - sz//2, sy - sz//2, sz, sz)
        pygame.draw.rect(screen, ENTRY_NEXT_OVERLAY, r, 0)
    for x,y in entry_back_spawns:
        sx, sy = world_to_screen(x,y)
        sz = max(2, int(ENTRY_MARK_SIZE * zoom))
        r = pygame.Rect(sx - sz//2, sy - sz//2, sz, sz)
        pygame.draw.rect(screen, ENTRY_BACK_OVERLAY, r, 0)

    # sample color from ORIGINAL bg (safe indexing)
    mx,my=pygame.mouse.get_pos(); wx,wy=screen_to_world(mx,my)
    if 0<=wx<WORLD_W and 0<=wy<WORLD_H:
        c = bg_world.get_at((clamp(int(wx),0,WORLD_W-1), clamp(int(wy),0,WORLD_H-1)))
    else:
        c = (0,0,0)
    pygame.draw.rect(screen, c, pygame.Rect(VIEW.right-54, VIEW.y+10, 44, 16))
    text(screen, f"{c[0]},{c[1]},{c[2]}", (VIEW.right-120, VIEW.y+10), C_TEXT_DIM, font_small)

    screen.set_clip(clip_old)

def draw_history_overlay():
    r = pygame.Rect(VIEW.x+40, VIEW.y+40, VIEW.w-80, VIEW.h-80)
    pygame.draw.rect(screen, (20,18,26), r, 0, border_radius=6)
    text(screen, "History  (view only)", (r.x+12, r.y+8))
    y=r.y+30
    for i,_entry in enumerate(reversed(undo_stack[-30:])):
        label=f"{len(undo_stack)-i:03d} • state"
        screen.blit(font_small.render(label, True, C_TEXT), (r.x+12, y)); y+=18

def draw_help_window():
    r = pygame.Rect(VIEW.x+30, VIEW.y+30, VIEW.w-60, VIEW.h-60)
    pygame.draw.rect(screen, (22,20,30), r, 0, border_radius=6)
    text(screen, "Help — Shortcuts & Features", (r.x+12, r.y+8))
    lines = [
        "Modes: PageUp=Edit (Move) · PageDown=Create (last tool)",
        "Tools: B=Brush · L=Line · Q=Bezier · O=Door► · U=Door◄ · Y=Entry► · M=Entry◄ · P=Spawn · H=Hand",
        "Door: click polygon; dbl-click/Enter to commit (bakes green/blue on Save)",
        "Spawn: click to place (red). Entry spawns: Y/M to add yellow/magenta pixels (baked on Save).",
        "Commit: Enter (current) · 0 (force straight) · Double-click",
        "Pan/Zoom: Space/MMB Drag · Wheel · 1/2/3 presets · Fit",
        "Brush: [ / ] size · S simplify  · G grid  · X/T symmetry  · E edges (lazy-build)",
        "Layers: ↑/↓ select · Ctrl+↑/↓ reorder walls · F2 rename · L lock · eye toggle",
        "Project: Ctrl+Shift+S save project · Ctrl+O load · Ctrl+S export PNG",
        "History: F9 overlay · Ctrl+Z / Ctrl+Y",
    ]
    y=r.y+32
    for ln in lines:
        screen.blit(font_small.render("• "+ln, True, C_TEXT), (r.x+12, y)); y+=18

# ---------- SAVE / LOAD ----------
def _mask_with_spawns_pixels(base_surf: pygame.Surface, spawn, entries_next, entries_back):
    # batch lock (faster)
    out = base_surf.copy()
    out.lock()
    if spawn:
        x, y = map(int, spawn)
        if 0 <= x < out.get_width() and 0 <= y < out.get_height():
            out.set_at((x, y), (255, 0, 0))
    for (x,y) in entries_next:
        xi, yi = int(x), int(y)
        if 0 <= xi < out.get_width() and 0 <= yi < out.get_height():
            out.set_at((xi, yi), ENTRY_NEXT_BAKE_COLOR)
    for (x,y) in entries_back:
        xi, yi = int(x), int(y)
        if 0 <= xi < out.get_width() and 0 <= yi < out.get_height():
            out.set_at((xi, yi), ENTRY_BACK_BAKE_COLOR)
    out.unlock()
    return out

def _bake_doors_fill(out_surf: pygame.Surface):
    for d in doors:
        if not d.get('visible', True): continue
        if len(d['pts']) >= 3:
            kind = d.get('kind','next')
            col = DOOR_NEXT_BAKE_COLOR if kind=='next' else DOOR_BACK_BAKE_COLOR
            pygame.draw.polygon(out_surf, col, d['pts'])
    return out_surf

def save_mask_png():
    path = next_asset_path()
    out_surf = _mask_with_spawns_pixels(mask_world, spawn_pos, entry_next_spawns, entry_back_spawns)
    out_surf = _bake_doors_fill(out_surf)
    pygame.image.save(out_surf, path)
    print(f"✅ Saved mask -> {path}")
    print("   baked: walls=white, spawn=red, door►=green, door◄=blue, entry►=yellow, entry◄=magenta")

def save_project():
    data = {
        "bg_path": BG_PATH,
        "world_size": [WORLD_W, WORLD_H],
        "strokes": strokes,
        "doors": doors,  # includes 'kind'
        "brush_w": brush_w,
        "preview_alpha": preview_alpha,
        "grid_on": grid_on, "grid_size": grid_size,
        "simplify_on": simplify_on, "sym_x": sym_x, "sym_y": sym_y,
        "spawn_pos": list(spawn_pos) if spawn_pos is not None else None,
        "entry_next_spawns": entry_next_spawns,
        "entry_back_spawns": entry_back_spawns,
    }
    out = filedialog.asksaveasfilename(title="Save Project", defaultextension=".xzenp",
                                       filetypes=[("Xzen Project",".xzenp")])
    if not out: return
    with open(out, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    print("💾 Project saved:", out)

def load_project():
    global BG_PATH, bg_world, mask_world, strokes, doors, brush_w, preview_alpha, grid_on, grid_size
    global simplify_on, sym_x, sym_y, sel_kind, sel_idx, spawn_pos
    global entry_next_spawns, entry_back_spawns, _scaled_mask_dirty, _last_zoom, _cached_bg, _cached_edges
    p = filedialog.askopenfilename(title="Load Project", filetypes=[("Xzen Project",".xzenp")])
    if not p: return
    with open(p, "r", encoding="utf-8") as f: data = json.load(f)
    path = data.get("bg_path", BG_PATH)
    if os.path.exists(path):
        BG_PATH = path
        bg_world = pygame.image.load(BG_PATH).convert()
        bg_world = pygame.transform.scale(bg_world, (WORLD_W, WORLD_H)).convert()
        _cached_bg = None; _cached_edges = None; _last_zoom = -1.0
    strokes = data.get("strokes", [])
    doors   = data.get("doors", [])
    for d in doors:
        if 'kind' not in d: d['kind'] = 'next'
    brush_w = data.get("brush_w", LINE_WIDTH_DEFAULT)
    preview_alpha = data.get("preview_alpha", 96)
    grid_on  = data.get("grid_on", False); grid_size = data.get("grid_size", 16)
    simplify_on = data.get("simplify_on", False)
    sym_x = data.get("sym_x", False); sym_y = data.get("sym_y", False)
    spawn_pos = tuple(data["spawn_pos"]) if data.get("spawn_pos") else None
    entry_next_spawns = [tuple(p) for p in data.get("entry_next_spawns", [])]
    entry_back_spawns = [tuple(p) for p in data.get("entry_back_spawns", [])]
    sel_kind, sel_idx = None, None
    update_mask()
    print("📂 Project loaded:", p)

def invert_mask():
    w,h = mask_world.get_size()
    for y in range(h):
        for x in range(w):
            r,g,b,*_ = mask_world.get_at((x,y))
            mask_world.set_at((x,y), (255-r,255-g,255-b))
    # mark dirty
    global _scaled_mask_dirty
    _scaled_mask_dirty = True

def clear_all():
    strokes.clear()
    doors.clear()
    entry_next_spawns.clear()
    entry_back_spawns.clear()
    global spawn_pos, _scaled_mask_dirty
    spawn_pos=None
    update_mask()
    _scaled_mask_dirty = True

def reorder_layer(delta):
    global sel_idx, sel_kind
    if sel_kind != "stroke": return
    if sel_idx is None or not strokes: return
    j = clamp(sel_idx+delta, 0, len(strokes)-1)
    if j == sel_idx: return
    strokes[sel_idx], strokes[j] = strokes[j], strokes[sel_idx]
    sel_idx = j; update_mask()

# ---------- MAIN ----------
def main():
    global tool, create_tool, mode, dragging, drag_started, last_mouse, sel_kind, sel_idx, ox, oy, last_click_time
    global space_pan, mmb_pan, brush_w, grid_on, preview_alpha, simplify_on, sym_x, sym_y, renaming, rename_buf
    global history_show, help_show, bezier_temp, zoom, show_edges, spawn_pos, points, door_points
    global entry_next_spawns, entry_back_spawns, edges_overlay, _cached_edges, _last_zoom

    def zoom_to(factor, anchor_screen=None):
        global zoom, ox, oy
        factor = clamp(factor, 0.5, 6.0)
        if anchor_screen is None: anchor_screen=(VIEW.centerx, VIEW.centery)
        ax_old, ay_old = anchor_screen
        wx,wy = screen_to_world(ax_old, ay_old)
        zoom = factor
        sx,sy = world_to_screen(wx,wy)
        ox += (ax_old - sx); oy += (ay_old - sy)

    def fit_to_view():
        zw=(VIEW.w-8)/WORLD_W; zh=(VIEW.h-8)/WORLD_H
        zoom_to(min(zw,zh), VIEW.center)

    fit_to_view()
    running=True
    while running:
        _dt = clock.tick(120)
        mx,my = pygame.mouse.get_pos()

        for e in pygame.event.get():
            if e.type == pygame.QUIT: running=False

            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: space_pan=True
            if e.type == pygame.KEYUP   and e.key == pygame.K_SPACE: space_pan=False; dragging=False; drag_started=False

            if e.type == pygame.MOUSEWHEEL and VIEW.collidepoint((mx,my)):
                if pygame.key.get_mods() & pygame.KMOD_ALT:
                    preview_alpha = clamp(preview_alpha + (10 if e.y>0 else -10), 10, 255)
                else:
                    if e.y>0: zoom_to(zoom*1.1,(mx,my))
                    elif e.y<0: zoom_to(zoom/1.1,(mx,my))

            if e.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if e.key == pygame.K_PAGEUP:   mode="edit"; tool=TOOL_MOVE
                if e.key == pygame.K_PAGEDOWN: mode="create"; tool=create_tool
                if e.key == pygame.K_b: create_tool=TOOL_BRUSH; mode="create"; tool=create_tool; points.clear()
                if e.key == pygame.K_l: create_tool=TOOL_LINE;  mode="create"; tool=create_tool; points.clear()
                if e.key == pygame.K_q: create_tool=TOOL_BEZ;   mode="create"; tool=create_tool; bezier_temp.clear()
                if e.key == pygame.K_o: create_tool=TOOL_DOOR_NEXT;  mode="create"; tool=create_tool; door_points.clear()
                if e.key == pygame.K_u: create_tool=TOOL_DOOR_BACK;  mode="create"; tool=create_tool; door_points.clear()
                if e.key == pygame.K_y: tool=TOOL_ENTRY_NEXT
                if e.key == pygame.K_m: tool=TOOL_ENTRY_BACK
                if e.key == pygame.K_p: tool=TOOL_SPAWN
                # move/hand
                if e.key == pygame.K_m: mode="edit"; tool=TOOL_MOVE  
                if e.key == pygame.K_h: tool=TOOL_HAND
                if e.key == pygame.K_1: zoom_to(0.75,(mx,my))
                if e.key == pygame.K_2: zoom_to(1.0,(mx,my))
                if e.key == pygame.K_3: zoom_to(2.0,(mx,my))
                if e.key == pygame.K_LEFTBRACKET:  brush_w = clamp(brush_w-1, 1, 128); update_mask()
                if e.key == pygame.K_RIGHTBRACKET: brush_w = clamp(brush_w+1, 1, 128); update_mask()
                if e.key == pygame.K_MINUS: preview_alpha = clamp(preview_alpha-10, 10, 255)
                if e.key == pygame.K_EQUALS: preview_alpha = clamp(preview_alpha+10, 10, 255)
                if e.key == pygame.K_g: grid_on = not grid_on
                if e.key == pygame.K_x: sym_x = not sym_x
                if e.key == pygame.K_t: sym_y = not sym_y  
                if e.key == pygame.K_s: simplify_on = not simplify_on
                if e.key == pygame.K_e:
                    show_edges = not show_edges
                    if show_edges and edges_overlay is None:
                        edges_overlay = compute_edges_surface(bg_world)
                        _cached_edges = None; _last_zoom = -1.0  # force rescale

                # selection cycle
                if e.key == pygame.K_UP and not (mods & pygame.KMOD_CTRL):
                    all_count = len(strokes) + len(doors)
                    if all_count:
                        if sel_kind is None or sel_idx is None:
                            sel_kind, sel_idx = ("stroke", 0) if len(strokes)>0 else ("door", 0)
                        else:
                            si = cast(int, sel_idx)
                            flat_index = (si if sel_kind=="stroke" else len(strokes)+si) - 1
                            if flat_index < 0: flat_index = all_count-1
                            if flat_index < len(strokes):
                                sel_kind, sel_idx = "stroke", flat_index
                            else:
                                sel_kind, sel_idx = "door", flat_index - len(strokes)
                if e.key == pygame.K_DOWN and not (mods & pygame.KMOD_CTRL):
                    all_count = len(strokes) + len(doors)
                    if all_count:
                        if sel_kind is None or sel_idx is None:
                            sel_kind, sel_idx = ("stroke", 0) if len(strokes)>0 else ("door", 0)
                        else:
                            si = cast(int, sel_idx)
                            flat_index = (si if sel_kind=="stroke" else len(strokes)+si) + 1
                            if flat_index >= all_count: flat_index = 0
                            if flat_index < len(strokes):
                                sel_kind, sel_idx = "stroke", flat_index
                            else:
                                sel_kind, sel_idx = "door", flat_index - len(strokes)

                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_UP: reorder_layer(-1)
                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_DOWN: reorder_layer(+1)

                # lock/rename
                if e.key == pygame.K_l and sel_kind is not None and sel_idx is not None and not renaming:
                    if sel_kind=="stroke":
                        strokes[cast(int, sel_idx)]['locked'] = not strokes[cast(int, sel_idx)].get('locked', False); update_mask()
                    else:
                        doors[cast(int, sel_idx)]['locked'] = not doors[cast(int, sel_idx)].get('locked', False)
                if e.key == pygame.K_F2 and sel_kind is not None and sel_idx is not None:
                    renaming = True
                    if sel_kind=="stroke":
                        rename_buf = strokes[cast(int, sel_idx)].get('name',"")
                    else:
                        rename_buf = doors[cast(int, sel_idx)].get('name',"")

                if renaming:
                    if e.key == pygame.K_RETURN and sel_kind is not None and sel_idx is not None:
                        if sel_kind=="stroke":
                            strokes[cast(int, sel_idx)]['name']=rename_buf
                        else:
                            doors[cast(int, sel_idx)]['name']=rename_buf
                        renaming=False
                    elif e.key == pygame.K_ESCAPE:
                        renaming=False
                    elif e.key == pygame.K_BACKSPACE:
                        rename_buf = rename_buf[:-1]
                    else:
                        ch = e.unicode
                        if ch.isprintable(): rename_buf += ch

                # delete / duplicate
                if e.key == pygame.K_DELETE and sel_kind is not None and sel_idx is not None:
                    push_undo()
                    if sel_kind=="stroke":
                        idx = cast(int, sel_idx)
                        strokes.pop(idx); update_mask()
                        if not strokes and doors: sel_kind, sel_idx = "door", 0
                        elif strokes: sel_idx = clamp(idx, 0, len(strokes)-1)
                        else: sel_kind, sel_idx = None, None
                    else:
                        idx = cast(int, sel_idx)
                        doors.pop(idx)
                        if doors: sel_idx = clamp(idx, 0, len(doors)-1)
                        elif strokes: sel_kind, sel_idx = "stroke", 0
                        else: sel_kind, sel_idx = None, None

                if e.key == pygame.K_d and not (mods & pygame.KMOD_CTRL) and sel_kind is not None and sel_idx is not None:
                    push_undo()
                    if sel_kind=="stroke":
                        idx = cast(int, sel_idx)
                        strokes.append(copy.deepcopy(strokes[idx])); sel_idx=len(strokes)-1; update_mask()
                    else:
                        idx = cast(int, sel_idx)
                        doors.append(copy.deepcopy(doors[idx])); sel_idx=len(doors)-1

                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_BACKSPACE:
                    push_undo(); clear_all()
                if (mods & pygame.KMOD_CTRL) and (mods & pygame.KMOD_SHIFT) and e.key == pygame.K_s: save_project()
                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_o: load_project()
                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_s: save_mask_png()

                # commit create tools
                if e.key in (pygame.K_RETURN,):
                    if mode=="create":
                        if create_tool==TOOL_BRUSH and len(points)>=2:
                            push_undo()
                            pts = points[:] + symmetry_mirror_pts(points)
                            commit_points('poly', pts)
                            points.clear()
                        if create_tool==TOOL_LINE and len(points)>=2:
                            push_undo()
                            pts = points[:] + symmetry_mirror_pts(points)
                            commit_points('straight_poly', pts)
                            points.clear()
                        if create_tool==TOOL_BEZ and len(bezier_temp)==3:
                            push_undo()
                            a,c,b = bezier_temp
                            samples=[]; steps=max(10,int(math.dist(a,b)/6))
                            for i in range(steps+1):
                                t=i/steps
                                x=int((1-t)**2*a[0]+2*(1-t)*t*c[0]+t**2*b[0])
                                y=int((1-t)**2*a[1]+2*(1-t)*t*c[1]+t**2*b[1])
                                samples.append((x,y))
                            samples += symmetry_mirror_pts(samples)
                            commit_points('bezier', samples)
                            bezier_temp.clear()
                        if create_tool in (TOOL_DOOR_NEXT, TOOL_DOOR_BACK) and len(door_points)>=3:
                            push_undo()
                            pts = door_points[:] + symmetry_mirror_pts(door_points)
                            kind = 'next' if create_tool==TOOL_DOOR_NEXT else 'back'
                            commit_door_points(pts, kind); door_points.clear()

                if e.key == pygame.K_0 and mode=="create" and points:
                    push_undo()
                    pts = points[:] + symmetry_mirror_pts(points)
                    commit_points('straight_poly', pts)
                    points.clear()
                if e.key == pygame.K_F9: history_show = not history_show
                if e.key == pygame.K_F1: help_show = not help_show
                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_z: do_undo()
                if (mods & pygame.KMOD_CTRL) and e.key == pygame.K_y: do_redo()

            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button==1 and SAVE_BTN_RECT.collidepoint((mx,my)): save_mask_png()

                if VIEW.collidepoint((mx,my)):
                    wx,wy = screen_to_world(mx,my)
                    if grid_on:
                        wx = round(wx / grid_size) * grid_size
                        wy = round(wy / grid_size) * grid_size

                    now = pygame.time.get_ticks()
                    dbl=(now-last_click_time)<=DOUBLE_CLICK_MS; last_click_time=now

                    if (space_pan and e.button==1) or e.button==2:
                        dragging=True; mmb_pan = (e.button==2); last_mouse=(mx,my); drag_started=False

                    elif tool==TOOL_SPAWN and e.button in (1,3,2):
                        if e.button==1:
                            push_undo(); spawn_pos = (int(wx), int(wy))
                            print("🎯 spawn set to:", spawn_pos)
                        else:
                            push_undo(); spawn_pos = None
                            print("🗑️ spawn cleared")

                    elif tool==TOOL_ENTRY_NEXT:
                        if e.button==1:
                            push_undo(); entry_next_spawns.append((int(wx),int(wy)))
                            print("➕ entry► (yellow) at:", entry_next_spawns[-1])
                        elif e.button in (3,2):
                            if entry_next_spawns:
                                sx,sy=int(wx),int(wy)
                                idx=min(range(len(entry_next_spawns)),
                                        key=lambda i:(entry_next_spawns[i][0]-sx)**2+(entry_next_spawns[i][1]-sy)**2)
                                if (entry_next_spawns[idx][0]-sx)**2+(entry_next_spawns[idx][1]-sy)**2 <= 36:
                                    push_undo(); entry_next_spawns.pop(idx)
                                    print("🗑️ removed nearest entry►")
                    elif tool==TOOL_ENTRY_BACK:
                        if e.button==1:
                            push_undo(); entry_back_spawns.append((int(wx),int(wy)))
                            print("➕ entry◄ (magenta) at:", entry_back_spawns[-1])
                        elif e.button in (3,2):
                            if entry_back_spawns:
                                sx,sy=int(wx),int(wy)
                                idx=min(range(len(entry_back_spawns)),
                                        key=lambda i:(entry_back_spawns[i][0]-sx)**2+(entry_back_spawns[i][1]-sy)**2)
                                if (entry_back_spawns[idx][0]-sx)**2+(entry_back_spawns[idx][1]-sy)**2 <= 36:
                                    push_undo(); entry_back_spawns.pop(idx)
                                    print("🗑️ removed nearest entry◄")

                    elif mode=="create" and create_tool==TOOL_BRUSH and e.button==1:
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT and points:
                            ax,ay=points[-1]
                            if abs(wx-ax)>abs(wy-ay): wy=ay
                            else: wx=ax
                        if dbl and len(points)>=2:
                            push_undo(); pts=points[:] + symmetry_mirror_pts(points)
                            commit_points('poly', pts); points.clear()
                        else:
                            points.append((int(wx),int(wy)))

                    elif mode=="create" and create_tool==TOOL_LINE and e.button==1:
                        if dbl and len(points)>=2:
                            push_undo(); pts=points[:] + symmetry_mirror_pts(points)
                            commit_points('straight_poly', pts); points.clear()
                        else:
                            points.append((int(wx),int(wy)))

                    elif mode=="create" and create_tool==TOOL_BEZ and e.button==1:
                        bezier_temp.append((int(wx),int(wy)))
                        if len(bezier_temp)==3:
                            push_undo()
                            a,c,b = bezier_temp
                            samples=[]; steps=max(10,int(math.dist(a,b)/6))
                            for i in range(steps+1):
                                t=i/steps
                                x=int((1-t)**2*a[0]+2*(1-t)*t*c[0]+t**2*b[0])
                                y=int((1-t)**2*a[1]+2*(1-t)*t*c[1]+t**2*b[1])
                                samples.append((x,y))
                            samples += symmetry_mirror_pts(samples)
                            commit_points('bezier', samples)
                            bezier_temp.clear()

                    elif mode=="create" and create_tool in (TOOL_DOOR_NEXT, TOOL_DOOR_BACK) and e.button==1:
                        if dbl and len(door_points)>=3:
                            push_undo()
                            pts = door_points[:] + symmetry_mirror_pts(door_points)
                            kind = 'next' if create_tool==TOOL_DOOR_NEXT else 'back'
                            commit_door_points(pts, kind); door_points.clear()
                        else:
                            door_points.append((int(wx),int(wy)))

                    elif tool==TOOL_MOVE and e.button==1:
                        didx = hit_test_doors((wx,wy))
                        sidx = hit_test_strokes((wx,wy))
                        if didx is not None and (not doors[didx].get('locked',False)):
                            sel_kind, sel_idx = "door", didx; dragging=True; last_mouse=(mx,my); drag_started=False
                        elif sidx is not None and (not strokes[sidx].get('locked',False)):
                            sel_kind, sel_idx = "stroke", sidx; dragging=True; last_mouse=(mx,my); drag_started=False

                # right panel interactions
                rows_local = draw_right_panel()  # draw now; reuse rows for hit tests
                for kind, i, rr, eye, lkr in rows_local:
                    if rr.collidepoint((mx,my)) and e.button==1:
                        sel_kind, sel_idx = kind, i
                    if eye.collidepoint((mx,my)) and e.button==1:
                        if kind=="stroke":
                            strokes[i]['visible']=not strokes[i].get('visible',True); update_mask()
                        else:
                            doors[i]['visible']=not doors[i].get('visible',True)
                    if lkr.collidepoint((mx,my)) and e.button==1:
                        if kind=="stroke":
                            strokes[i]['locked']=not strokes[i].get('locked',False)
                        else:
                            doors[i]['locked']=not doors[i].get('locked',False)

                # left toolbar
                btns,_=draw_left_toolbar((mx,my))
                for key, rect, _ in btns:
                    if rect.collidepoint((mx,my)) and e.button==1:
                        if key in (TOOL_BRUSH,TOOL_LINE,TOOL_BEZ,TOOL_DOOR_NEXT,TOOL_DOOR_BACK):
                            create_tool=key; mode="create"; tool=create_tool
                            points.clear(); bezier_temp.clear(); door_points.clear()
                        elif key==TOOL_MOVE:
                            mode="edit"; tool=TOOL_MOVE
                        elif key==TOOL_HAND: tool=TOOL_HAND
                        elif key==TOOL_SPAWN: tool=TOOL_SPAWN
                        elif key==TOOL_ENTRY_NEXT: tool=TOOL_ENTRY_NEXT
                        elif key==TOOL_ENTRY_BACK: tool=TOOL_ENTRY_BACK
                        elif key=="fit":
                            fit_to_view()
                        elif key=="del" and sel_kind is not None and sel_idx is not None:
                            push_undo()
                            if sel_kind=="stroke":
                                idx = cast(int, sel_idx)
                                strokes.pop(idx); update_mask()
                                if not strokes and doors: sel_kind, sel_idx = "door", 0
                                elif strokes: sel_idx = clamp(idx, 0, len(strokes)-1)
                                else: sel_kind, sel_idx = None, None
                            else:
                                idx = cast(int, sel_idx)
                                doors.pop(idx)
                                if doors: sel_idx = clamp(idx, 0, len(doors)-1)
                                elif strokes: sel_kind, sel_idx = "stroke", 0
                                else: sel_kind, sel_idx = None, None
                        elif key=="dup" and sel_kind is not None and sel_idx is not None:
                            push_undo()
                            if sel_kind=="stroke":
                                idx = cast(int, sel_idx)
                                strokes.append(copy.deepcopy(strokes[idx])); sel_idx=len(strokes)-1; update_mask()
                            else:
                                idx = cast(int, sel_idx)
                                doors.append(copy.deepcopy(doors[idx])); sel_idx=len(doors)-1
                        elif key=="clear":
                            push_undo(); clear_all()
                        break

            if e.type == pygame.MOUSEBUTTONUP:
                if e.button == 2: mmb_pan = False
                dragging=False; drag_started=False

            if e.type == pygame.MOUSEMOTION and dragging:
                if (tool==TOOL_HAND) or space_pan or mmb_pan:
                    ox += (e.pos[0]-last_mouse[0]); oy += (e.pos[1]-last_mouse[1]); last_mouse=e.pos
                elif tool==TOOL_MOVE and sel_kind is not None and sel_idx is not None:
                    dx=(e.pos[0]-last_mouse[0])/zoom; dy=(e.pos[1]-last_mouse[1])/zoom
                    if abs(dx) or abs(dy):
                        if not drag_started:
                            push_undo(); drag_started=True
                        if sel_kind=="stroke" and not strokes[cast(int, sel_idx)].get('locked',False):
                            st=strokes[cast(int, sel_idx)]
                            st['pts']=[(int(x+dx), int(y+dy)) for (x,y) in st['pts']]
                            update_mask(); last_mouse=e.pos
                        elif sel_kind=="door" and not doors[cast(int, sel_idx)].get('locked',False):
                            d=doors[cast(int, sel_idx)]
                            d['pts']=[(int(x+dx), int(y+dy)) for (x,y) in d['pts']]
                            last_mouse=e.pos

        screen.fill(C_BG)
        draw_save_row()
        btns, hovered = draw_left_toolbar((mx,my))
        _rows = draw_right_panel()
        draw_viewport()
        if hovered: draw_tooltip(hovered[0], hovered[1])
        if history_show: draw_history_overlay()
        if help_show: draw_help_window()
        wx,wy = screen_to_world(mx,my)
        msg = f"{int(wx)}, {int(wy)}"
        if mode=="create":
            if create_tool in (TOOL_BRUSH, TOOL_LINE) and points: msg += f"  pts:{len(points)}"
            if create_tool==TOOL_BEZ and bezier_temp: msg += f"  bez:{len(bezier_temp)}"
            if create_tool in (TOOL_DOOR_NEXT, TOOL_DOOR_BACK) and door_points: msg += f"  door_pts:{len(door_points)}"
        draw_status(msg)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("💥 Error:", ex)
        pygame.quit(); sys.exit(1)

# Unfinished but functional