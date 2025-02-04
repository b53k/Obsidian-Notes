/*
THIS IS A GENERATED/BUNDLED FILE BY ESBUILD
if you want to view the source, please visit the github repository of this plugin
*/

var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// main.ts
var main_exports = {};
__export(main_exports, {
  default: () => MinWidthPlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian = require("obsidian");
var DEFAULT_SETTINGS = {
  maxWidthPercent: "88%",
  defaultMinWidth: "40rem",
  minWidthOfViewType: {}
};
var DATA_VIEW_TYPE = "data-min-width-plugin-view-type";
var CLASS_ACTIVE = "min-width-plugin-active";
function defaultSetting(value, defaultValue) {
  if (value === void 0 || value === null || value.trim() === "") {
    return defaultValue;
  }
  return value;
}
function findResizeEl(el) {
  const parent = el.parentElement;
  if (parent !== null && parent.hasClass("workspace-tab-container")) {
    const grandParent = parent.parentElement;
    if (grandParent !== null && grandParent.hasClass("workspace-tabs")) {
      return grandParent;
    }
  }
  return el;
}
var MinWidthPlugin = class extends import_obsidian.Plugin {
  async onload() {
    await this.loadSettings();
    const head = window.activeDocument.head;
    this.styleTag = head.createEl("style");
    this.styleTag.id = "min-width-plugin-style";
    this.injectStyles();
    this.addSettingTab(new MinWidthSettingTab(this.app, this));
    this.registerEvent(this.app.workspace.on("active-leaf-change", (0, import_obsidian.debounce)((leaf) => this.onActiveLeafChange(leaf), 200)));
  }
  onActiveLeafChange(leaf) {
    if (leaf === null) {
      return;
    }
    const leafEl = leaf.view.containerEl.parentElement;
    if (leafEl === null) {
      return;
    }
    this.removeClassesFrom(leafEl.doc.body);
    const resizeEl = findResizeEl(leafEl);
    resizeEl.addClass(CLASS_ACTIVE);
    const dataType = leaf.view.containerEl.getAttribute("data-type");
    resizeEl.setAttr(DATA_VIEW_TYPE, dataType);
    const resizeParentEl = resizeEl.parentElement;
    if (resizeParentEl !== null && resizeParentEl.hasClass("mod-horizontal")) {
      resizeParentEl.addClass(CLASS_ACTIVE);
      resizeParentEl.setAttr(DATA_VIEW_TYPE, dataType);
    }
  }
  onunload() {
    this.styleTag.innerText = "";
    this.styleTag.remove();
    this.removeClassesFrom(window.activeDocument.body);
    window.activeDocument.body.findAll(`.mod-horizontal[${DATA_VIEW_TYPE}], .workspace-leaf[${DATA_VIEW_TYPE}]`).forEach((el) => el.setAttr(DATA_VIEW_TYPE, null));
  }
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
    this.injectStyles();
  }
  injectStyles() {
    const { maxWidthPercent, defaultMinWidth, minWidthOfViewType } = this.settings;
    const cssStyles = `
			.mod-root .${CLASS_ACTIVE} {
				min-width: min(${maxWidthPercent}, ${defaultMinWidth});
			}
			${Object.entries(minWidthOfViewType).map(([viewType, minWidth]) => `
			.mod-root .${CLASS_ACTIVE}[${DATA_VIEW_TYPE}="${viewType}"] {
				min-width: min(${maxWidthPercent}, ${minWidth});
			}
			`).join(" ")}
		`.trim().replace(/[\r\n\s]+/g, " ");
    this.styleTag.innerText = cssStyles;
  }
  removeClassesFrom(rootEl) {
    rootEl.findAll(`.${CLASS_ACTIVE}`).forEach((el) => el.removeClass(CLASS_ACTIVE));
  }
};
var MinWidthSettingTab = class extends import_obsidian.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Min Width Settings" });
    new import_obsidian.Setting(containerEl).setName("Max Width Percent").setDesc("Set the upper bound of the min width to the percentage of the whole editing area.").addText((text) => text.setPlaceholder(DEFAULT_SETTINGS.maxWidthPercent).setValue(this.plugin.settings.maxWidthPercent).onChange(async (value) => {
      this.plugin.settings.maxWidthPercent = defaultSetting(value, DEFAULT_SETTINGS.maxWidthPercent);
      await this.plugin.saveSettings();
    }));
    new import_obsidian.Setting(containerEl).setName("Min Width").setDesc("Set the minimum width of the active pane. The format is a number followed by a unit, e.g. 40rem. The unit can be px, rem, em, vw, vh, vmin, vmax, %.").addText((text) => text.setPlaceholder(DEFAULT_SETTINGS.defaultMinWidth).setValue(this.plugin.settings.defaultMinWidth).onChange(async (value) => {
      this.plugin.settings.defaultMinWidth = defaultSetting(value, DEFAULT_SETTINGS.defaultMinWidth);
      await this.plugin.saveSettings();
    }));
    containerEl.createEl("h3", {
      text: "View Types Settings"
    });
    for (const [viewType, minWidth] of Object.entries(this.plugin.settings.minWidthOfViewType)) {
      new import_obsidian.Setting(containerEl).setName(viewType).setDesc('The same format as "Min Width"').addText((text) => text.setValue(minWidth).onChange(async (value) => {
        this.plugin.settings.minWidthOfViewType[viewType] = defaultSetting(value, DEFAULT_SETTINGS.defaultMinWidth);
        await this.plugin.saveSettings();
      })).addButton((button) => {
        button.setIcon("trash").onClick(async () => {
          delete this.plugin.settings.minWidthOfViewType[viewType];
          this.display();
          await this.plugin.saveSettings();
        });
      });
    }
    let newViewType = "";
    new import_obsidian.Setting(containerEl).addText((text) => text.setValue(newViewType).onChange(async (value) => newViewType = value.trim())).addButton((button) => button.setButtonText("Add View Type").onClick(async () => {
      if (newViewType !== "" && !(newViewType in this.plugin.settings.minWidthOfViewType)) {
        this.plugin.settings.minWidthOfViewType[newViewType] = "";
        this.display();
      }
    }));
  }
};
