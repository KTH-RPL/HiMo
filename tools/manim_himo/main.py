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
            and min(p3[0], p4[0]) <= x <= max(p3[0], p4[0])
            and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1])
        ):
            return np.array([x, y, 0])
        else:
            return None

    def lidar_ray(self, lidar, inner_rect_points, start=0, end=360):
        rays = []
        dots = []
        lidar_position = lidar.get_center()

        for angle in range(start, end, 10):  # Every 10 degrees, a ray is emitted
            ray_dir = np.array([np.cos(angle * DEGREES), np.sin(angle * DEGREES), 0])
            intersections = []

            for i in range(len(inner_rect_points)):
                p1 = inner_rect_points[i]
                p2 = inner_rect_points[(i + 1) % len(inner_rect_points)]
                intersection_point = self.get_line_intersection(lidar_position, lidar_position + 10 * ray_dir, p1, p2)
                if intersection_point is not None:
                    intersections.append(intersection_point)

            if intersections:
                # Find the closest intersection point
                closest_point = min(intersections, key=lambda point: np.linalg.norm(point - lidar_position))
                ray = Line(start=lidar_position, end=closest_point, color=YELLOW, buff=0).add_tip(tip_length=0.1).set_opacity(0.5)
                dot = Dot(closest_point, color=BLUE)
                rays.append(ray)
                dots.append(dot)

        return rays, dots

    def scan_looks(self, rec, lidar, dots, rays, relative_obj):
        rec.set_stroke(opacity=0.3)
        rec.set_fill(opacity=0)
        copy_dots = [dot.copy() for dot in dots]
        copy_dots2 = [dot.copy() for dot in dots]
        group2 = VGroup(rec, lidar, *copy_dots, *(rays.copy()))
        group1 = VGroup(rec.copy(), lidar.copy(), *[dot.scale(1.5) for dot in copy_dots2])
        self.play(Create(group2))
        self.play(group2.animate.scale(0.3).next_to(relative_obj, DOWN))

        self.play(Create(group1))
        self.play(group1.animate.scale(0.3).next_to(group2, DOWN))
        return group1
    
    def construct(self):
        # Create objects
        ego_car = Rectangle(width=1, height=2, color=DARK_BROWN)#.scale(0.5)
        lidar = Circle(radius=0.1, color=BLUE).move_to(ego_car.get_center()) # .scale(0.5)
        rec1 = Rectangle(width=7, height=7)
        rec2 = Rectangle(width=6, height=6) #  color=BLACK
        static_wall = Difference(rec1, rec2, color=GREY, fill_opacity=0.9)

        # Simulate the moving ego car and lidar
        o_ego_car = ego_car.copy().move_to(ORIGIN)
        o_lidar = lidar.copy().move_to(ORIGIN)
        o_staticwall = static_wall.copy().move_to(ORIGIN)
        o_rec2 = rec2.copy().move_to(ORIGIN)
        
        # Initial Setup
        # 
        # self.add(ego_car)
        # self.add(lidar)
        text= Text("This is our ego car", font_size = 18)
        text.next_to(ego_car, UR)
        self.play(Create(ego_car), Write(text))
        self.wait(0.5)


        self.remove(text)
        text = Text("We have a lidar sensor", font_size = 18)
        text.next_to(ego_car, RIGHT)
        self.play(Write(text), Create(lidar))

        self.play(VGroup(ego_car, lidar).animate.scale(0.5))
        self.remove(text)
        text = Text("Now we add static environment", font_size = 18).next_to(ego_car, UP)
        self.play(Write(text))
        self.remove(text)
        self.add(static_wall)






        # # Generate and animate the rays and dots in the static case
        text = Text("Static Case", font_size=18).to_corner(UP, buff=0.5)
        self.play(Create(text))
        rays, dots = self.lidar_ray(lidar, rec2.get_vertices())
        for ray, dot in zip(rays, dots):
            self.play(Create(ray), Create(dot), run_time=0.05)

        

        copy_dots = [dot.copy() for dot in dots]
        copy_rays = [ray.copy() for ray in rays]
        group = VGroup(static_wall, ego_car, lidar, *[dot for dot in dots], *[ray for ray in rays])
        self.play(group.animate.scale(0.3).to_corner(UL))
        self.remove(text)
        text = Text("One Scan Point Cloud", font_size=18).to_corner(UP)
        self.play(Create(text))
        static_ego = self.scan_looks(o_rec2.copy(), o_lidar.copy(), copy_dots, copy_rays, group)
        self.remove(text)

        m_staticwall = o_staticwall.copy()
        moving_group = VGroup(o_ego_car.copy(), o_lidar.copy())
        self.add(m_staticwall)
        total_rays, total_dots = [], []
        moving_raysdots = VGroup(o_rec2.copy().set_stroke(opacity=0.3).set_fill(opacity=0), o_lidar.copy())
        moving_dots = VGroup(o_rec2.copy().set_stroke(opacity=0.3).set_fill(opacity=0), o_lidar.copy())
        text = Text("If we are moving", font_size=18).to_corner(UP)
        self.play(Create(text))
        for inter_ in range(0, 360, 30):
            rays, dots = self.lidar_ray(moving_group[1], o_rec2.get_vertices(), start=inter_, end=inter_+30)
            self.add(moving_group)
            tmp_raydots = VGroup()
            tmp_dots = VGroup()
            for ray, dot in zip(rays, dots):
                self.play(Create(ray), Create(dot), run_time=0.05)
                total_rays.append(ray)
                total_dots.append(dot)
                tmp_raydots.add(ray.copy(), dot.copy())
                tmp_dots.add(dot.copy().scale(1.5))
            lidar_start_position = moving_group[1].get_center()
            tmp_raydots.shift(-lidar_start_position)
            tmp_dots.shift(-lidar_start_position)
            moving_raysdots.add(tmp_raydots)
            moving_dots.add(tmp_dots)
            moving_group.move_to(UP*inter_/360)
            self.wait(0.1)
        self.remove(text)

        text = Text("One Scan Point Cloud", font_size=18).to_corner(UP)
        self.play(Create(text))

        moving_group.add(*total_rays, *total_dots, m_staticwall)
        self.play(moving_group.animate.scale(0.3).to_corner(UR))
        self.play(Create(moving_raysdots))
        self.play(moving_raysdots.animate.scale(0.3).next_to(moving_group, DOWN))
        self.play(Create(moving_dots))
        self.play(moving_dots.animate.scale(0.3).next_to(moving_raysdots, DOWN))
        self.remove(text)

        # text = Text("Comparison", font_size=18).to_corner(RIGHT)
        # self.play(Create(text))
        self.play(static_ego.animate.scale(1.8).move_to(UP*2))
        self.play(moving_dots.animate.scale(1.8).next_to(static_ego, DOWN))
        # self.remove(text)

        self.wait(1)