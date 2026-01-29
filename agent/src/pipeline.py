"""
Generative UI Pipeline

端到端 Pipeline：从标注数据生成 A2UI JSON
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any
from tqdm import tqdm

from .data_loader import DataLoader, Recommendation, SceneConfig
from .llm_client import LLMClient
from .component_selector import ComponentSelector
from .props_filler import PropsFiller
from .schema_validator import SchemaValidator


class GenerativeUIPipeline:
    """Generative UI Pipeline"""

    def __init__(
        self,
        data_path: str,
        output_path: str,
        participant: str = "P1_YuePan",
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
    ):
        self.data_loader = DataLoader(data_path, participant)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 初始化 LLM 客户端
        self.llm_client = LLMClient(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

        # 初始化组件选择器和填充器
        self.component_selector = ComponentSelector(self.llm_client)
        self.props_filler = PropsFiller(self.llm_client)
        self.validator = SchemaValidator()

    def process_recommendation(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> dict[str, Any] | None:
        """处理单条推荐，生成 A2UI 组件"""
        try:
            # Step 1: 选择组件
            selection = self.component_selector.select_component(recommendation, scene)
            component_type = selection.get("selected_component")

            if not component_type:
                print(f"  ⚠️ 无法选择组件: {recommendation.content[:30]}...")
                return None

            # Step 2: 填充 props
            component = self.props_filler.generate_component(
                component_type, recommendation
            )

            # Step 3: 验证
            is_valid, error = self.validator.validate_component(component)
            if not is_valid:
                print(f"  ⚠️ 组件验证失败: {error}")
                return None

            # 添加选择信息
            component["metadata"]["selection"] = {
                "reasoning": selection.get("reasoning", ""),
                "confidence": selection.get("confidence", 0.0),
            }

            return component

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            return None

    def run(
        self,
        scenes: list[str] | None = None,
        limit: int = 50,
        save_individual: bool = True,
    ) -> list[dict[str, Any]]:
        """运行 Pipeline"""
        if scenes is None:
            scenes = ["navigation", "shopping"]

        print(f"🚀 Starting Generative UI Pipeline")
        print(f"   Scenes: {scenes}")
        print(f"   Limit: {limit}")
        print()

        results = []
        scene_results = {scene: [] for scene in scenes}

        # 迭代处理
        data_iter = list(self.data_loader.iter_mvp_data(scenes, limit))

        for recommendation, scene in tqdm(data_iter, desc="Processing"):
            component = self.process_recommendation(recommendation, scene)
            if component:
                results.append(component)
                scene_results[scene.name].append(component)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存汇总文件
        summary_file = self.output_path / f"a2ui_components_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 按场景保存
        for scene_name, components in scene_results.items():
            if components:
                scene_file = self.output_path / f"a2ui_{scene_name}_{timestamp}.json"
                with open(scene_file, "w", encoding="utf-8") as f:
                    json.dump(components, f, ensure_ascii=False, indent=2)

        # 打印统计
        print()
        print(f"✅ Pipeline completed!")
        print(f"   Total components: {len(results)}")
        for scene_name, components in scene_results.items():
            print(f"   - {scene_name}: {len(components)}")
        print(f"   Output: {summary_file}")

        return results


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Smart Glasses Generative UI Pipeline")
    parser.add_argument(
        "--data-path",
        default="/home/v-tangxin/GUI/data/ego-dataset",
        help="数据集路径",
    )
    parser.add_argument(
        "--output-path",
        default="/home/v-tangxin/GUI/agent/output",
        help="输出路径",
    )
    parser.add_argument(
        "--participant",
        default="P1_YuePan",
        help="被试 ID",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["navigation", "shopping"],
        help="要处理的场景",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="最大处理数量",
    )

    args = parser.parse_args()

    pipeline = GenerativeUIPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        participant=args.participant,
    )

    pipeline.run(scenes=args.scenes, limit=args.limit)


if __name__ == "__main__":
    main()
