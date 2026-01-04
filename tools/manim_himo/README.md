HiMo annimation video
---

We use manim to create the animation video for project website. The code is under this folder (`tools/manim_himo`).

I create a new conda python environment for manim package, please following their [installation guide](https://docs.manim.community/en/stable/installation.html) to setup your environment.

It's also my first time to use this tools, I refer to their documentation and discuss with ChatGPT to create the animation video. I would strongly recommend to read their [QuickStart Guide](https://docs.manim.community/en/stable/tutorials/quickstart.html) first to get familiar with the basic usage.

## Usage

To render our video, you can run the following command:
```bash
manim -pql scene.py
manim -pql single_dynamic.py
manim -qh dynamic.py
manim -qh multi_v2.py
```

And each of them will generate a mp4 video file under `media/videos/` folder. Here is a example folder structure from my local machine:
```
➜  manim_himo git:(main) tree -L 2
.
├── dynamic.py
├── __init__.py
├── main.py
├── media
│   ├── images
│   │   ├── dynamic
│   │   ├── multi_v2
│   │   ├── scene
│   │   └── single_dynamic
│   ├── texts
│   └── videos
│   └── videos
│       ├── dynamic
│       │   ├── 1080p60
│       │   └── 480p15
│       ├── multi_v2
│       │   ├── 1080p60
│       │   └── 480p15
│       ├── scene
│       │   ├── 1080p60
│       │   └── 480p15
│       └── single_dynamic
│           └── 480p15
├── multi_dynamic.py
├── multi_v2.py
├── scene.py
└── single_dynamic.py
```

Feel free to explore more, and happy manimming! 🎬