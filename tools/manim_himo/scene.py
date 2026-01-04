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

class EgoMotion(Scene):

    def ego_assets(self):
        ego_car = Rectangle(width=1, height=2, color=DARK_BROWN)#.scale(0.5).move_to(UP)
        lidar = Circle(radius=0.1, color=BLUE).scale(2.5)
        return VGroup(ego_car, lidar)
    
    def display_text(self, text, font_size=18):
        text = Text(text, font_size = font_size)
        self.play(Write(text))
        self.wait(1)
        self.remove(text)

    def lidar_ray(self, sensor_pose, rect_points, start=-90, end=270, angle_interval=5):
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
            else:
                ray = None
                dot = None
            rays.append(ray)
            dots.append(dot)
            angles.append(angle)

        return rays, dots, angles

    def scene_group(self, ego, objs, others):
        pass
        # self.play(Create(group))
        # self.wait(1)
        # self.play(FadeOut(group))

    def construct(self):
        Gego = self.ego_assets()
        text = Text("This is our ego vehicle", font_size = 18).next_to(Gego[0], UP)
        self.play(Create(Gego[0]), Write(text))
        text2 = Text("With a LiDAR sensor", font_size = 18, color=BLUE).next_to(Gego[0], UP)
        self.play(FadeOut(text), Write(text2), Gego[1].animate.scale(0.8))
        self.play(FadeOut(text2), Gego.animate.shift(UP).scale(0.5))

        # Create objects
        obj2 = Rectangle(width=2, height=2).next_to(Gego[0], DOWN*10) # , color=BLACK
        obj1 = Rectangle(width=1, height=2).next_to(Gego[0], LEFT*8+UP*4) # , color=BLACK
        obj3 = Rectangle(width=1, height=2).next_to(Gego[0], RIGHT*8+DOWN*8) # , color=BLACK
        text = Text("Add other objects", font_size = 18).next_to(Gego[0], UP)
        self.play(Write(text), Create(obj1), Create(obj2), Create(obj3))
        self.remove(text)
        Graydots = VGroup()
        rays, dots, angles = [], [], []
        for obj in [obj1, obj2, obj3]:
            ray, dot, angle = self.lidar_ray(Gego[1].get_center(), obj.get_vertices())
            rays.extend(ray)
            dots.extend(dot)
            angles.extend(angle)

        rays, dots, angles = sort_angle(rays, dots, angles)

        # self.
        Graydots_delete = VGroup()
        for ray, dot, angle in zip(rays, dots, angles):
            start_pos = Gego[1].get_center()
            ray_dir = np.array([np.cos(angle* DEGREES), np.sin(angle* DEGREES), 0]) 
            ray_ = Line(start=start_pos, end=start_pos+ray_dir*MAX_RAY_LENGTH, buff=0).set_opacity(0.2)
            ray_.set_color(Gego[1].get_color())
            self.add(ray_)
            if ray is not None:
                self.play(Create(ray), Create(dot), run_time=0.1)
                Graydots_delete.add(ray, dot)
                Graydots.add(ray.copy().set_opacity(0.1), dot) # .copy().scale(1.5)
            self.remove(ray_)
        
        
        # self.play(Create(text))
        text = Text("If we are moving", font_size = 18).next_to(Gego[0], UP*2+RIGHT)
        self.play(FadeOut(Graydots_delete), Write(text))
        Graydots_moving = VGroup()
        ego_frame_ego = Gego.copy()
        ego_frame_raydots = VGroup()
        for inter_ in range(0, 360, 30):
            tmp_raydots = VGroup()
            tmp_dots = VGroup()
            rays, dots, angles = [], [], []
            ego_pos = Gego[1].get_center()
            for obj in [obj1, obj2, obj3]:
                ray, dot, angle = self.lidar_ray(ego_pos, obj.get_vertices(), start=inter_-90, end=inter_+30-90)
                rays.extend(ray)
                dots.extend(dot)
                angles.extend(angle)
            for ray, dot in zip(rays, dots):
                if ray is not None:
                    self.play(Create(ray), Create(dot), run_time=0.01, rate_func=linear)
                    Graydots_moving.add(ray, dot)
                    tmp_raydots.add(ray.copy(), dot.copy())
                    tmp_dots.add(dot.copy())
            
            ego_frame_raydots.add(tmp_raydots.shift(-ego_pos).set_opacity(0.1), tmp_dots.shift(-ego_pos))
            self.play(Gego.animate.shift(UP*inter_*0.2/360), runtime=0.001, rate_func=linear)
            # self.wait(0.1)


        self.play(FadeOut(Graydots_moving), FadeOut(Gego), FadeOut(text))
        Gobj = VGroup(obj1, obj2, obj3)
        Gobj.set_opacity(0.1)
        text = Text("Ego-motion distorted data", font_size = 18).next_to(ego_frame_ego, UP+RIGHT)
        text2 = Text("Centered to the ego vehicle", font_size = 18).next_to(text, DOWN)
        Gtext = VGroup(text, text2)
        self.play(Write(Gtext))
        self.play(Create(ego_frame_ego), Create(ego_frame_raydots.shift(UP)))
        self.remove(Gtext)

        
        wo_ego_comp = VGroup(ego_frame_ego, ego_frame_raydots, Gobj)
        ego_comp = VGroup(ego_frame_ego.copy(), Graydots, Gobj.copy())
        # 计算两个对象的总宽度
        total_width = wo_ego_comp.width + ego_comp.width

        # 计算第一个对象的中心点应该在的位置（场景中心的左侧）
        wo_ego_comp_target_position = ORIGIN + LEFT * (total_width / 2 - wo_ego_comp.width / 2) - LEFT

        # 计算第二个对象的中心点应该在的位置（场景中心的右侧）
        ego_comp_target_position = ORIGIN + RIGHT * (total_width / 2 - ego_comp.width / 2) + LEFT

        # 移动两个对象到计算好的位置
        self.play(wo_ego_comp.animate.scale(0.6).move_to(wo_ego_comp_target_position))
        self.play(FadeIn(Text("w/o ego-motion comp.", font_size=18).next_to(wo_ego_comp, DOWN)))
        self.play(ego_comp.animate.scale(0.6).move_to(ego_comp_target_position))
        self.play(FadeIn(Text("w. ego-motion comp.", font_size=18).next_to(ego_comp, DOWN)))  

        self.wait(2)