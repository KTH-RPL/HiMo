"""
# Created: 2024-08-25 22:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of HiMo (https://github.com/KTH-RPL/HiMo) projects.
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Draw animation video we present.
"""

from manim import *
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from manim_himo import get_line_intersection, MAX_RAY_LENGTH, sort_angle

class SingleLiDAR(Scene):

    def ego_assets(self):
        ego_car = Rectangle(width=1, height=2, color=DARK_BROWN).scale(0.5).move_to(UP)
        lidar = Circle(radius=0.1, color=BLUE).scale(0.5).move_to(ego_car.get_center())
        return VGroup(ego_car, lidar)
    
    def display_text(self, text, font_size=18):
        text = Text(text, font_size = font_size)
        self.play(Write(text))
        self.wait(1)
        self.remove(text)

    def lidar_ray(self, sensor_pose, rect_points, start=-90, end=270, angle_interval=10):
        rays, dots, angles = [], [], []
        for angle in range(start, end, angle_interval): 
            ray_dir = np.array([np.cos(angle * DEGREES), np.sin(angle * DEGREES), 0])
            intersections = []

            for i in range(len(rect_points)):
                p1 = rect_points[i]
                p2 = rect_points[(i + 1) % len(rect_points)]
                intersection_point = get_line_intersection(sensor_pose, sensor_pose + MAX_RAY_LENGTH * ray_dir, p1, p2)
                if intersection_point is not None:
                    intersections.append(intersection_point)

            if intersections:
                # Find the closest intersection point
                closest_point = min(intersections, key=lambda point: np.linalg.norm(point - sensor_pose))
                ray = Line(start=sensor_pose, end=closest_point, color=YELLOW, buff=0).add_tip(tip_length=0.1).set_opacity(0.5)
                dot = Dot(closest_point, color=BLUE, radius=0.12)
                rays.append(ray)
                dots.append(dot)
                angles.append(angle)
        return rays, dots, angles

    def scene_group(self, speed=0.1):
        Gego = self.ego_assets()
        obj1 = Rectangle(width=1, height=2).next_to(Gego[0], LEFT*8) # , color=BLACK
        obj2 = Rectangle(width=2, height=3).next_to(Gego[0], DOWN*8) # , color=BLACK
        obj3 = Rectangle(width=1, height=2).next_to(Gego[0], RIGHT*8) # , color=BLACK
        self.add(Gego, obj1, obj2, obj3)
        # Simulate the moving obj1 and obj2
        inter_set =5
        Graydots = VGroup()
        Gdots = VGroup()
        for inter_ in range(0, 360, inter_set):
            rays, dots, angles = [], [], []
            for obj in [obj1, obj2, obj3]:
                ray, dot, angle = self.lidar_ray(Gego[1].get_center(), obj.get_vertices(), start=inter_-90, end=inter_+inter_set-90)
                rays.extend(ray)
                dots.extend(dot)
                angles.extend(angle)
            # order rays and dots based on the angle
            if len(rays) == 0:
                continue
            rays, dots, angles = sort_angle(rays, dots, angles)
            start_pos = Gego[1].get_center()
            ray_dir = np.array([np.cos((inter_-90) * DEGREES), np.sin((inter_+inter_set-90) * DEGREES), 0])
            redray = Line(start=start_pos, end=start_pos+ray_dir*100, color=RED, buff=0).add_tip(tip_length=0.1).set_opacity(0.5)
            self.add(redray)
            for ray, dot in zip(rays, dots):
                self.play(Create(ray), Create(dot), run_time=0.05)
                Gdots.add(dot)
                Graydots.add(ray)
            # self.wait(0.1)
            self.play(obj1.animate.shift(DOWN*speed*inter_set/360), \
                      obj2.animate.shift(UP*speed*inter_set/360), \
                      obj3.animate.shift(UP*speed*inter_set/360), run_time=0.05)
            self.remove(redray)

        for obj in [obj1, obj2, obj3]:
            obj.set_opacity(0.1)
        self.play(FadeOut(Graydots), FadeOut(Gdots), FadeOut(obj1), FadeOut(obj2), FadeOut(obj3), FadeOut(Gego))
        return VGroup(Gego, obj1, obj2, obj3, Gdots)
    
    def construct(self):

        slow = self.scene_group(speed=0.1)
        self.remove(slow)
        fast = self.scene_group(speed=2.2)
        self.remove(fast)
        total_width = slow.width + fast.width

        # 计算第一个对象的中心点应该在的位置（场景中心的左侧）
        slow_target_position = ORIGIN + LEFT * (total_width / 2 - slow.width / 2) - LEFT

        # 计算第二个对象的中心点应该在的位置（场景中心的右侧）
        fast_target_position = ORIGIN + RIGHT * (total_width / 2 - fast.width / 2) + LEFT

        # 移动两个对象到计算好的位置
        self.play(slow.animate.scale(0.6).move_to(slow_target_position))
        self.play(FadeIn(Text("Slow-speed objects", font_size=18).next_to(slow, DOWN)))
        self.play(fast.animate.scale(0.6).move_to(fast_target_position))
        self.play(FadeIn(Text("High-speed objects", font_size=18).next_to(fast, DOWN)))  


        self.wait(2)