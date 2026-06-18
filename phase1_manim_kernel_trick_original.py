from manim import *
import numpy as np
from utils.data_generator import generate_concentric_circles

class SVMKernelTrick3D(ThreeDScene):
    def construct(self):
        # ----------------------------------------------------------------------
        # 1. 設置標題與背景說明
        # ----------------------------------------------------------------------
        title = Text("SVM 核心技巧: 2D 到 3D 空間對應", font_size=28, color=WHITE)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        intro_text = Text(
            "同心圓資料點在 2D 平面中無法以直線做線性分割。",
            font_size=16,
            color=LIGHT_GRAY
        )
        intro_text.next_to(title, DOWN)
        self.add_fixed_in_frame_mobjects(intro_text)
        
        # ----------------------------------------------------------------------
        # 2. 設置座標軸與 2D 點陣散佈圖
        # ----------------------------------------------------------------------
        # 相機初始化為俯視 (2D 平面視角)
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-2.0, 2.0, 0.5],
            y_range=[-2.0, 2.0, 0.5],
            z_range=[0.0, 2.5, 0.5],
            x_length=5.5,
            y_length=5.5,
            z_length=4.0
        )
        axes.center()
        
        # 產生同心圓資料點 (樣本點 50 個，避免動畫過於擁擠)
        X, y = generate_concentric_circles(noise=0.04, n_samples=50, random_seed=42)
        
        # 建立 Manim 中的 Dot3D/Dot 資料點群組
        dots = VGroup()
        for idx in range(len(X)):
            x_coord, y_coord = X[idx]
            label = y[idx]
            
            # Label = 0 -> 內圈 (藍色), Label = 1 -> 外圈 (紅色)
            color = BLUE if label == 0 else RED
            dot_pos = axes.c2p(x_coord, y_coord, 0.0)
            dot = Dot(point=dot_pos, radius=0.08, color=color)
            dots.add(dot)
            
        self.play(Create(axes), Write(dots), run_time=2)
        self.wait(1.5)
        
        # ----------------------------------------------------------------------
        # 3. 嘗試在 2D 中做線性切割
        # ----------------------------------------------------------------------
        failing_line = Line(
            start=axes.c2p(-1.8, -1.2, 0.0),
            end=axes.c2p(1.8, 1.2, 0.0),
            color=GRAY,
            stroke_width=3
        )
        failing_text = Text(
            "在 2D 平面中，任意直線都會造成嚴重的分類錯誤。",
            font_size=15,
            color=RED_B
        )
        failing_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(failing_text)
        
        self.play(Create(failing_line))
        self.wait(1)
        self.play(FadeOut(failing_line), FadeOut(failing_text))
        
        # ----------------------------------------------------------------------
        # 4. 展示 3D 特徵映射函數
        # ----------------------------------------------------------------------
        formula_text = Text(
            "phi(x, y) = (x, y, x² + y²)",
            font_size=20,
            color=YELLOW
        )
        formula_text.next_to(intro_text, DOWN)
        self.add_fixed_in_frame_mobjects(formula_text)
        
        mapping_explain = Text(
            "藉由平方距離，將二維坐標點映射（Lifting）至三維空間。",
            font_size=15,
            color=LIGHT_GRAY
        )
        mapping_explain.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(mapping_explain)
        
        self.play(Write(formula_text), Write(mapping_explain))
        self.wait(2)
        self.play(FadeOut(mapping_explain))
        
        # ----------------------------------------------------------------------
        # 5. 轉換相機鏡頭至 3D 視角
        # ----------------------------------------------------------------------
        self.move_camera(phi=70 * DEGREES, theta=-55 * DEGREES, run_time=2)
        self.wait(1)
        
        # ----------------------------------------------------------------------
        # 6. 拉升資料點至 3D 空間 (Lifting)
        # ----------------------------------------------------------------------
        lifting_text = Text(
            "沿 Z 軸將資料點拉升：z = x² + y²",
            font_size=15,
            color=LIGHT_GRAY
        )
        lifting_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(lifting_text)
        self.play(Write(lifting_text))
        
        lift_animations = []
        for idx, dot in enumerate(dots):
            x_coord, y_coord = X[idx]
            z_coord = x_coord**2 + y_coord**2
            target_pos = axes.c2p(x_coord, y_coord, z_coord)
            lift_animations.append(dot.animate.move_to(target_pos))
            
        self.play(*lift_animations, run_time=3)
        self.wait(1.5)
        self.play(FadeOut(lifting_text))
        
        # ----------------------------------------------------------------------
        # 7. 渲染 3D 拋物曲面
        # ----------------------------------------------------------------------
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-1.6, 1.6],
            v_range=[-1.6, 1.6],
            fill_opacity=0.3,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        surface_text = Text(
            "此時所有資料點皆完美貼合在 3D 拋物面上。",
            font_size=15,
            color=LIGHT_GRAY
        )
        surface_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(surface_text)
        
        self.play(Create(paraboloid), Write(surface_text), run_time=2.5)
        self.wait(2)
        self.play(FadeOut(surface_text))
        
        # ----------------------------------------------------------------------
        # 8. 引入劃分超平面 (Separating Hyperplane)
        # ----------------------------------------------------------------------
        # 內圈點半徑為 ~0.4 -> z 高度約 0.16
        # 外圈點半徑為 ~1.0 -> z 高度約 1.0
        # 超平面設定在 z = 0.5 可以達到完美分割
        plane_height = 0.5
        
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, plane_height),
            u_range=[-1.6, 1.6],
            v_range=[-1.6, 1.6],
            fill_opacity=0.45,
            checkerboard_colors=[ORANGE, YELLOW]
        )
        
        plane_text = Text(
            f"引入一個 3D 水平超平面 (z = {plane_height}) 來將兩個類別分開。",
            font_size=15,
            color=LIGHT_GRAY
        )
        plane_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(plane_text)
        
        self.play(Create(hyperplane), Write(plane_text), run_time=2)
        self.wait(2)
        self.play(FadeOut(plane_text))
        
        # ----------------------------------------------------------------------
        # 9. 環繞相機旋轉
        # ----------------------------------------------------------------------
        orbit_text = Text(
            "旋轉視角以展示超平面如何完美切分兩種類別。",
            font_size=15,
            color=LIGHT_GRAY
        )
        orbit_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(orbit_text)
        self.play(Write(orbit_text))
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(orbit_text))
        
        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=1)
        
        # ----------------------------------------------------------------------
        # 10. 計算相交圓並投影回 2D
        # ----------------------------------------------------------------------
        circle_radius = np.sqrt(plane_height)
        
        # 3D 空間超平面高度的交界圓 (z = 0.5)
        intersection_circle = ParametricFunction(
            lambda t: axes.c2p(circle_radius * np.cos(t), circle_radius * np.sin(t), plane_height),
            t_range=[0, 2 * np.pi],
            color=YELLOW,
            stroke_width=5
        )
        
        # 投影至地面上的交界圓 (z = 0.0)
        projected_circle = ParametricFunction(
            lambda t: axes.c2p(circle_radius * np.cos(t), circle_radius * np.sin(t), 0.0),
            t_range=[0, 2 * np.pi],
            color=YELLOW,
            stroke_width=5
        )
        
        intersection_text = Text(
            "相交處會形成一個 3D 圓形邊界，並將其投影回 2D 地面。",
            font_size=15,
            color=LIGHT_GRAY
        )
        intersection_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(intersection_text)
        
        self.play(Create(intersection_circle), Write(intersection_text), run_time=2)
        self.wait(1.5)
        
        # 動畫：複製圓線並拉降投影至 2D 地面
        self.play(
            TransformFromCopy(intersection_circle, projected_circle),
            run_time=2
        )
        self.wait(1.5)
        self.play(FadeOut(intersection_text))
        
        # ----------------------------------------------------------------------
        # 11. 返回 2D 俯視視角展示最終結果
        # ----------------------------------------------------------------------
        final_text = Text(
            "投影至 2D 得到的圓形非線性決策邊界：x² + y² = c",
            font_size=15,
            color=YELLOW
        )
        final_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(final_text)
        
        self.move_camera(
            phi=0,
            theta=-90 * DEGREES,
            added_anims=[FadeOut(paraboloid), FadeOut(hyperplane), FadeOut(intersection_circle)],
            run_time=2.5
        )
        
        # 將資料點拉降回原點 z = 0
        return_animations = []
        for idx, dot in enumerate(dots):
            x_coord, y_coord = X[idx]
            target_pos = axes.c2p(x_coord, y_coord, 0.0)
            return_animations.append(dot.animate.move_to(target_pos))
            
        self.play(*return_animations, Write(final_text), run_time=2)
        self.wait(3)
