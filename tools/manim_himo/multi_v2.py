from manim import *
# Mobject.set_default(color=BLACK)
# config.background_color = WHITE
class HiMo_Static(Scene):
    def get_line_intersection(self, p1, p2, p3, p4):
        A1 = p2[1] - p1[1]
        B1 = p1[0] - p2[0]
        C1 = A1 * p1[0] + B1 * p1[1]

        A2 = p4[1] - p3[1]
        B2 = p3[0] - p4[0]
        C2 = A2 * p3[0] + B2 * p3[1]

        determinant = A1 * B2 - A2 * B1

        if determinant == 0:
            return None  # The lines are parallel or coincident

        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant

        if (
            min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])
            and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])
            and min(p3[0], p4[0])-0.01 <= x <= max(p3[0], p4[0])+0.1
            and min(p3[1], p4[1])-0.01 <= y <= max(p3[1], p4[1])+0.1
        ):
            return np.array([x, y, 0])
        else:
            return None
    def lidar_ray(self, lidar, inner_rect_points, start=-90, end=270, angle_set=10):
        rays = []
        dots = []
        lidar_position = lidar.get_center()
        angles = []
        for angle in range(start, end, angle_set): 
            ray_dir = np.array([np.cos(angle * DEGREES), np.sin(angle * DEGREES), 0])
            closest_intersection = None
            min_distance = 10000
            for i in range(len(inner_rect_points)):
                p1 = inner_rect_points[i]
                p2 = inner_rect_points[(i + 1) % len(inner_rect_points)]
                intersection_point = self.get_line_intersection(lidar_position, lidar_position + ray_dir*100, p1, p2)
                if intersection_point is not None:
                    distance = np.linalg.norm(intersection_point - lidar_position)
                    # print(i, distance)
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = intersection_point
            
            if closest_intersection is not None:
                ray = Line(start=lidar_position, end=closest_intersection, color=YELLOW, buff=0).add_tip(tip_length=0.1).set_opacity(0.5)
                dot = Dot(closest_intersection, color=BLUE, radius=0.12)
                rays.append(ray)
                dots.append(dot)
                angles.append(angle)
                # print()
        return rays, dots, angles

    def raydots(self, objs, lidar, start=-90, end=270, angle_set=10):
        # Generate rays and dots
        rays, dots, angles = [], [], []
        for obj in objs:
            ray, dot, angle = self.lidar_ray(lidar, obj.get_vertices(), start=start, end=end, angle_set=angle_set)
            rays.extend(ray)
            dots.extend(dot)
            angles.extend(angle)

        if len(rays) == 0:
            return [], []
        
        combined = sorted(zip(angles, rays, dots), key=lambda x: x[0])
        angles, rays, dots = zip(*combined)
        return rays, dots
    
    def ego_assets(self):
        ego_car = Rectangle(width=1, height=2, color=DARK_BROWN)#.scale(0.5)#.move_to(UP)
        lidar1 = Circle(radius=0.1, color=BLUE)#.scale(0.5).move_to(ego_car.get_center())
        lidar2 = Circle(radius=0.2, color=GREEN)#.scale(0.5).move_to(ego_car.get_center())
        return VGroup(ego_car, lidar1, lidar2)
    
    def scene_group(self, speed=0.1, skip=False):
        Gego = self.ego_assets()
        if not skip:
            self.add(Gego[0])
            text2 = Text("With two LiDAR sensors", font_size = 18).next_to(Gego[0], UP)
            self.play(Write(text2), run_time=1.0)
            self.play(Gego[1].animate.scale(0.8))
            self.play(Gego[2].animate.scale(0.8))
            self.play(FadeOut(text2), Gego.animate.shift(UP).scale(0.5))
        else:
            Gego[1].scale(0.8)
            Gego[2].scale(0.8)
            Gego.shift(UP).scale(0.5)
            self.add(Gego)
        ego_car = Gego[0]
        obj2 = Rectangle(width=2, height=2).next_to(ego_car, DOWN*8) # , color=BLACK
        obj1 = Rectangle(width=1, height=2).next_to(ego_car, LEFT*8+UP*8) # , color=BLACK
        obj3 = Rectangle(width=1, height=2).next_to(ego_car, RIGHT*8+DOWN*8) # , color=BLACK
        # Initial Setup
        self.add(obj1, obj2, obj3)

        Graydots = VGroup()
        Gdots = VGroup()
        inter_set=5
        for inter_ in range(0, 360, inter_set):
            start_pos = Gego[1].get_center()
            ray_dir = np.array([np.cos((inter_-90) * DEGREES), np.sin((inter_+inter_set-90) * DEGREES), 0])
            ray1_ = Line(start=start_pos, end=start_pos+ray_dir*100, color=BLUE, buff=0).set_opacity(0.2)
            ray_dir = np.array([np.cos((inter_+90) * DEGREES), np.sin((inter_+inter_set+90) * DEGREES), 0])
            ray2_ = Line(start=start_pos, end=start_pos+ray_dir*100, color=GREEN, buff=0).set_opacity(0.2)
            ray = VGroup(ray1_, ray2_)
            self.add(ray)
            rays1, dots1 = self.raydots([obj1, obj2, obj3], Gego[1], start=inter_-90, end=inter_+inter_set-90)
            rays2, dots2 = self.raydots([obj1, obj2, obj3], Gego[2], start=inter_+90, end=inter_+inter_set+90)
            for i in range(max(len(rays1), len(rays2))):
                if i < len(rays1) and i < len(rays2):
                    dots1[i].scale(0.8)
                    dots2[i].scale(0.8).set_color(GREEN)
                    self.play(Create(rays1[i]), Create(dots1[i]), Create(rays2[i]), Create(dots2[i]), run_time=0.05)
                    Graydots.add(rays1[i], rays2[i])
                    Gdots.add(dots1[i], dots2[i])
                elif i < len(rays1):
                    dots1[i].scale(0.8)
                    self.play(Create(rays1[i]), Create(dots1[i]), run_time=0.05)
                    Graydots.add(rays1[i])
                    Gdots.add(dots1[i])
                elif i < len(rays2):
                    dots2[i].scale(0.8).set_color(GREEN)
                    self.play(Create(rays2[i]), Create(dots2[i]), run_time=0.05)
                    Graydots.add(rays2[i])
                    Gdots.add(dots2[i])

            self.play(obj1.animate.shift(DOWN*speed*inter_set/360), \
                      obj2.animate.shift(UP*speed/2*inter_set/360), \
                      obj3.animate.shift(UP*speed*inter_set/360), run_time=0.05)
            self.remove(ray)

        for obj in [obj1, obj2, obj3]:
            obj.set_opacity(0.1)

        self.play(FadeOut(Graydots), FadeOut(Gdots), FadeOut(obj1), FadeOut(obj2), FadeOut(obj3), FadeOut(Gego))
        return VGroup(Gego, obj1, obj2, obj3, Gdots)
    
    def construct(self):
        slow = self.scene_group(speed=0.1)
        self.remove(slow)
        fast = self.scene_group(speed=2, skip=True)
        self.remove(fast)
        total_width = slow.width + fast.width

        slow_target_position = ORIGIN + LEFT * (total_width / 2 - slow.width / 2) - LEFT
        fast_target_position = ORIGIN + RIGHT * (total_width / 2 - fast.width / 2) + LEFT

        self.play(slow.animate.scale(0.6).move_to(slow_target_position))
        self.play(FadeIn(Text("Slow-speed objects", font_size=18).next_to(slow, DOWN)))
        self.play(fast.animate.scale(0.6).move_to(fast_target_position))
        self.play(FadeIn(Text("High-speed objects", font_size=18).next_to(fast, DOWN)))  

        self.wait(2)