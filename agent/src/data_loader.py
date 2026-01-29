"""
数据加载器

读取 annotations_2.2.json 并按场景分类
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

# Import unified schema (single source of truth)
from .schema import Recommendation, SceneConfig, get_allowed_components


@dataclass
class TimeSegment:
    """时间段标注"""
    start_time: float
    end_time: float
    recommendations: list[Recommendation] = field(default_factory=list)
    accepted_recommendations: list[Recommendation] = field(default_factory=list)


class DataLoader:
    """数据加载器"""

    def __init__(self, base_path: str, participant: str = "P1_YuePan"):
        self.base_path = Path(base_path)
        self.participant = participant
        self.annotation_path = self.base_path / "data" / participant / "annotation"

        # 场景配置 (P1_YuePan)
        self.scenes = {
            "navigation": SceneConfig(
                name="navigation",
                allowed_components=["map_card", "ar_label", "direction_arrow", "comparison_card"],
                start_time=726.1,
                end_time=1533.3,
            ),
            "shopping": SceneConfig(
                name="shopping",
                allowed_components=["comparison_card", "nutrition_card", "price_calculator", "ar_label"],
                start_time=3307.3,
                end_time=4703.1,
            ),
        }

    def load_annotations(self, filename: str | None = None) -> list[TimeSegment]:
        """加载 annotation_2.2 文件"""
        if filename is None:
            # 查找 annotation_2.2 文件
            pattern = f"annotation_2.2_{self.participant}*.json"
            files = list(self.annotation_path.glob(pattern))
            if not files:
                # 尝试不同的命名模式
                pattern = f"annotations_2.2_{self.participant}*.json"
                files = list(self.annotation_path.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No annotation_2.2 file found in {self.annotation_path}")
            filepath = files[0]
        else:
            filepath = self.annotation_path / filename

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        rec_counter = 0
        for item in data:
            segment = TimeSegment(
                start_time=item["start_time"],
                end_time=item["end_time"],
            )

            # 解析推荐列表
            for rec in item.get("recommendation_list", []):
                rec_counter += 1
                segment.recommendations.append(Recommendation(
                    id=f"rec_{rec_counter}",
                    type=rec["type"],
                    content=rec["content"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                ))

            # 解析接受的推荐
            for rec in item.get("accepted_recommendation_list", []):
                rec_counter += 1
                segment.accepted_recommendations.append(Recommendation(
                    id=f"rec_{rec_counter}",
                    type=rec["type"],
                    content=rec.get("original_content", rec.get("content", "")),
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    object_list=rec.get("object_list"),
                    is_accepted=True,
                ))

            segments.append(segment)

        return segments

    def classify_scene(self, time: float) -> str | None:
        """根据时间判断场景类型"""
        for scene_name, scene in self.scenes.items():
            if scene.start_time <= time <= scene.end_time:
                return scene_name
        return None

    def get_scene_recommendations(
        self,
        scene_name: str,
        accepted_only: bool = True,
    ) -> list[tuple[Recommendation, SceneConfig]]:
        """获取指定场景的推荐"""
        if scene_name not in self.scenes:
            raise ValueError(f"Unknown scene: {scene_name}")

        scene = self.scenes[scene_name]
        segments = self.load_annotations()
        results = []

        for segment in segments:
            # 检查时间段是否与场景重叠
            if segment.end_time < scene.start_time or segment.start_time > scene.end_time:
                continue

            if accepted_only:
                for rec in segment.accepted_recommendations:
                    results.append((rec, scene))
            else:
                for rec in segment.recommendations:
                    results.append((rec, scene))

        return results

    def iter_mvp_data(
        self,
        scenes: list[str] | None = None,
        limit: int = 50,
    ) -> Iterator[tuple[Recommendation, SceneConfig]]:
        """迭代 MVP 数据"""
        if scenes is None:
            scenes = ["navigation", "shopping"]

        count = 0
        for scene_name in scenes:
            for rec, scene in self.get_scene_recommendations(scene_name, accepted_only=True):
                if count >= limit:
                    return
                yield rec, scene
                count += 1
