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
 * Badge component for status indicators.
 *
 * Variants: info, success, warning, error
 */
@customElement("a2ui-badge")
export class Badge extends Root {
  @property({ type: String })
  accessor variant: "info" | "success" | "warning" | "error" = "info";

  @property({ type: String })
  accessor text: string = "";

  static styles = [
    structuralStyles,
    css`
      :host {
        display: inline-flex;
      }

      .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        line-height: 1;
      }

      .badge.info {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
      }

      .badge.success {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
      }

      .badge.warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
      }

      .badge.error {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
      }
    `,
  ];

  render() {
    const classes = {
      badge: true,
      [this.variant]: true,
    };

    return html`<span class=${classMap(classes)}>
      <slot>${this.text}</slot>
    </span>`;
  }
}
