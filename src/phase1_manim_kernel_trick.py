from manim import *
import numpy as np
from src.utils.data_generator import generate_concentric_circles

class SVMKernelTrick3D(ThreeDScene):
    def construct(self):
        # ----------------------------------------------------------------------
        # 1. 設置日系星空背景與飄落粒子 (Starry Night & Sakura Particles)
        # ----------------------------------------------------------------------
        self.camera.background_color = "#0a0b16"  # 深邃的深夜藍靛色
        
        # 產生背景中的繁星與微弱漂浮粒子
        np.random.seed(123)  # 固定隨機種子
        particles = VGroup()
        for _ in range(45):
            pos = np.array([
                np.random.uniform(-7.5, 7.5),
                np.random.uniform(-4.5, 4.5),
                np.random.uniform(-1.0, 5.0)
            ])
            color = np.random.choice(["#ffb7c5", "#a5f3fc", "#ffffff", "#fef08a"])
            size = np.random.uniform(0.015, 0.04)
            particle = Dot(
                point=pos,
                radius=size,
                color=color,
                fill_opacity=np.random.uniform(0.2, 0.65)
            )
            particles.add(particle)
        
        self.add(particles)
        
        # 自訂色調定義
        SAKURA_PINK = "#ffb7c5"    # 櫻花粉 (類別 0 - 內圈)
        SKY_BLUE = "#a5f3fc"       # 晴空藍 (類別 1 - 外圈)
        GOLD_YELLOW = "#facc15"    # 耀金黃 (決策邊界)
        SLATE_GRAY = "#475569"     # 板岩灰 (座標軸)
        
        # ----------------------------------------------------------------------
        # 2. 設置標題與背景說明 (精緻字體設計)
        # ----------------------------------------------------------------------
        title = Text("SVM 核心技巧: 2D 到 3D 空間對應", font_size=28, color=WHITE, weight=BOLD)
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
        # 3. 設置座標軸與 2D 點陣散佈圖
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
        
        # 產生同心圓資料點 (樣本點 30 個，避免動畫過於擁擠)
        X, y = generate_concentric_circles(noise=0.04, n_samples=30, random_seed=42)
        
        # 建立 Manim 中的 Dot 資料點群組 (櫻花粉與晴空藍)
        dots = VGroup()
        for idx in range(len(X)):
            x_coord, y_coord = X[idx]
            label = y[idx]
            
            # Label = 0 -> 內圈 (櫻花粉), Label = 1 -> 外圈 (晴空藍)
            color = SAKURA_PINK if label == 0 else SKY_BLUE
            dot_pos = axes.c2p(x_coord, y_coord, 0.0)
            dot = Dot(point=dot_pos, radius=0.085, color=color)
            dots.add(dot)
            
        self.play(Create(axes), Write(dots), run_time=2)
        self.wait(1.5)
        
        # ----------------------------------------------------------------------
        # 4. 嘗試在 2D 中做線性切割
        # ----------------------------------------------------------------------
        failing_line = Line(
            start=axes.c2p(-1.8, -1.2, 0.0),
            end=axes.c2p(1.8, 1.2, 0.0),
            color=SLATE_GRAY,
            stroke_width=3
        )
        failing_text = Text(
            "在 2D 平面中，任意直線都會造成嚴重的分類錯誤。",
            font_size=15,
            color="#f87171"  # 珊瑚紅珊瑚色
        )
        failing_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(failing_text)
        
        self.play(Create(failing_line))
        self.wait(1)
        self.play(FadeOut(failing_line), FadeOut(failing_text))
        
        # ----------------------------------------------------------------------
        # 5. 展示 3D 特徵映射函數
        # ----------------------------------------------------------------------
        formula_text = Text(
            "phi(x, y) = (x, y, x² + y²)",
            font_size=20,
            color=GOLD_YELLOW
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
        # 6. 轉換相機鏡頭至 3D 視角 (運鏡更為溫柔緩慢)
        # ----------------------------------------------------------------------
        self.move_camera(phi=70 * DEGREES, theta=-55 * DEGREES, run_time=2.5)
        self.wait(0.5)
        
        # ----------------------------------------------------------------------
        # 7. 拉升資料點至 3D 空間 (Lifting)
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
            
        self.play(*lift_animations, run_time=3.5)
        self.wait(1.5)
        self.play(FadeOut(lifting_text))
        
        # ----------------------------------------------------------------------
        # 8. 渲染 3D 拋物曲面 (改用日系極光淡粉藍漸層感)
        # ----------------------------------------------------------------------
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-1.6, 1.6],
            v_range=[-1.6, 1.6],
            resolution=(16, 16),
            fill_opacity=0.35,
            checkerboard_colors=["#1e1b4b", "#0284c7"]  # 深靛色到晴空天藍
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
        # 9. 引入劃分超平面 (透光暖橙色超平面)
        # ----------------------------------------------------------------------
        plane_height = 0.5
        
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, plane_height),
            u_range=[-1.6, 1.6],
            v_range=[-1.6, 1.6],
            resolution=(4, 4),
            fill_opacity=0.4,
            checkerboard_colors=["#b45309", "#d97706"]  # 溫暖的日落橙橘色
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
        # 10. 環繞相機旋轉 (電影感橫向緩慢運鏡)
        # ----------------------------------------------------------------------
        orbit_text = Text(
            "旋轉視角以展示超平面如何完美切分兩種類別。",
            font_size=15,
            color=LIGHT_GRAY
        )
        orbit_text.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(orbit_text)
        self.play(Write(orbit_text))
        
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4.5)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(orbit_text))
        
        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=1)
        
        # ----------------------------------------------------------------------
        # 11. 計算相交圓並投影回 2D
        # ----------------------------------------------------------------------
        circle_radius = np.sqrt(plane_height)
        
        # 3D 空間交界圓 (耀金黃)
        intersection_circle = ParametricFunction(
            lambda t: axes.c2p(circle_radius * np.cos(t), circle_radius * np.sin(t), plane_height),
            t_range=[0, 2 * np.pi],
            color=GOLD_YELLOW,
            stroke_width=6
        )
        
        # 投影至地面上的交界圓 (耀金黃)
        projected_circle = ParametricFunction(
            lambda t: axes.c2p(circle_radius * np.cos(t), circle_radius * np.sin(t), 0.0),
            t_range=[0, 2 * np.pi],
            color=GOLD_YELLOW,
            stroke_width=6
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
        
        self.play(
            TransformFromCopy(intersection_circle, projected_circle),
            run_time=2
        )
        self.wait(1.5)
        self.play(FadeOut(intersection_text))
        
        # ----------------------------------------------------------------------
        # 12. 返回 2D 俯視視角展示最終結果
        # ----------------------------------------------------------------------
        final_text = Text(
            "投影至 2D 得到的圓形非線性決策邊界：x² + y² = c",
            font_size=15,
            color=GOLD_YELLOW
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
        self.wait(3.5)
