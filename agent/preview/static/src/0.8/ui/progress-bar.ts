/*
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

import { html, css } from "lit";
import { customElement, property } from "lit/decorators.js";
import { Root } from "./root.js";
import { classMap } from "lit/directives/class-map.js";
import { structuralStyles } from "./styles.js";

/**
 * ProgressBar component for displaying progress.
 *
 * Variants: default, slim
 */
@customElement("a2ui-progressbar")
export class ProgressBar extends Root {
  @property({ type: String })
  accessor variant: "default" | "slim" = "default";

  @property({ type: Number })
  accessor value: number = 0;

  @property({ type: Number })
  accessor max: number = 100;

  static styles = [
    structuralStyles,
    css`
      :host {
        display: block;
        width: 100%;
      }

      .progress-container {
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
        position: relative;
      }

      .progress-container.default {
        height: 8px;
        border-radius: 8px;
      }

      .progress-container.slim {
        height: 4px;
        border-radius: 4px;
      }

      .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        transition: width 0.3s ease;
      }

      .progress-container.default .progress-fill {
        border-radius: 8px;
      }

      .progress-container.slim .progress-fill {
        border-radius: 4px;
      }
    `,
  ];

  render() {
    const percent = this.max > 0 ? (this.value / this.max) * 100 : 0;
    const classes = {
      "progress-container": true,
      [this.variant]: true,
    };

    return html`
      <div class=${classMap(classes)}>
        <div class="progress-fill" style="width: ${percent}%"></div>
      </div>
    `;
  }
}
