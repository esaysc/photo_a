// vite.config.js
import { defineConfig, loadEnv } from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/vite/dist/node/index.js";
import path2 from "path";

// vite/plugins/index.js
import vue from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/@vitejs/plugin-vue/dist/index.mjs";

// vite/plugins/auto-import.js
import autoImport from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/unplugin-auto-import/dist/vite.js";
function createAutoImport() {
  return autoImport({
    imports: [
      "vue",
      "vue-router",
      "pinia"
    ],
    dts: false
  });
}

// vite/plugins/svg-icon.js
import { createSvgIconsPlugin } from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/vite-plugin-svg-icons/dist/index.mjs";
import path from "path";
function createSvgIcon(isBuild) {
  return createSvgIconsPlugin({
    iconDirs: [path.resolve(process.cwd(), "src/assets/icons/svg")],
    symbolId: "icon-[dir]-[name]",
    svgoOptions: isBuild
  });
}

// vite/plugins/compression.js
import compression from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/vite-plugin-compression/dist/index.mjs";
function createCompression(env) {
  const { VITE_BUILD_COMPRESS } = env;
  const plugin = [];
  if (VITE_BUILD_COMPRESS) {
    const compressList = VITE_BUILD_COMPRESS.split(",");
    if (compressList.includes("gzip")) {
      plugin.push(
        compression({
          ext: ".gz",
          deleteOriginFile: false
        })
      );
    }
    if (compressList.includes("brotli")) {
      plugin.push(
        compression({
          ext: ".br",
          algorithm: "brotliCompress",
          deleteOriginFile: false
        })
      );
    }
  }
  return plugin;
}

// vite/plugins/setup-extend.js
import setupExtend from "file:///I:/projects/java-vue/RuoYi-Vue-master/ruoyi-ui/node_modules/unplugin-vue-setup-extend-plus/dist/vite.js";
function createSetupExtend() {
  return setupExtend({});
}

// vite/plugins/index.js
function createVitePlugins(viteEnv, isBuild = false) {
  const vitePlugins = [vue()];
  vitePlugins.push(createAutoImport());
  vitePlugins.push(createSetupExtend());
  vitePlugins.push(createSvgIcon(isBuild));
  isBuild && vitePlugins.push(...createCompression(viteEnv));
  return vitePlugins;
}

// vite.config.js
var __vite_injected_original_dirname = "I:\\projects\\java-vue\\RuoYi-Vue-master\\ruoyi-ui";
var baseUrl = "http://localhost:8080";
var vite_config_default = defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, process.cwd());
  const { VITE_APP_ENV } = env;
  return {
    // 部署生产环境和开发环境下的URL。
    // 默认情况下，vite 会假设你的应用是被部署在一个域名的根路径上
    // 例如 https://www.ruoyi.vip/。如果应用被部署在一个子路径上，你就需要用这个选项指定这个子路径。例如，如果你的应用被部署在 https://www.ruoyi.vip/admin/，则设置 baseUrl 为 /admin/。
    base: VITE_APP_ENV === "production" ? "/" : "/",
    plugins: createVitePlugins(env, command === "build"),
    resolve: {
      // https://cn.vitejs.dev/config/#resolve-alias
      alias: {
        // 设置路径
        "~": path2.resolve(__vite_injected_original_dirname, "./"),
        // 设置别名
        "@": path2.resolve(__vite_injected_original_dirname, "./src")
      },
      // https://cn.vitejs.dev/config/#resolve-extensions
      extensions: [".mjs", ".js", ".ts", ".jsx", ".tsx", ".json", ".vue"]
    },
    // 打包配置
    build: {
      // https://vite.dev/config/build-options.html
      sourcemap: command === "build" ? false : "inline",
      outDir: "dist",
      assetsDir: "assets",
      chunkSizeWarningLimit: 2e3,
      rollupOptions: {
        output: {
          chunkFileNames: "static/js/[name]-[hash].js",
          entryFileNames: "static/js/[name]-[hash].js",
          assetFileNames: "static/[ext]/[name]-[hash].[ext]"
        }
      }
    },
    // vite 相关配置
    server: {
      port: 80,
      host: true,
      open: true,
      proxy: {
        // https://cn.vitejs.dev/config/#server-proxy
        "/dev-api": {
          target: baseUrl,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/dev-api/, "")
        },
        // springdoc proxy
        "^/v3/api-docs/(.*)": {
          target: baseUrl,
          changeOrigin: true
        }
      }
    },
    css: {
      postcss: {
        plugins: [
          {
            postcssPlugin: "internal:charset-removal",
            AtRule: {
              charset: (atRule) => {
                if (atRule.name === "charset") {
                  atRule.remove();
                }
              }
            }
          }
        ]
      }
    }
  };
});
export {
  baseUrl,
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiLCAidml0ZS9wbHVnaW5zL2luZGV4LmpzIiwgInZpdGUvcGx1Z2lucy9hdXRvLWltcG9ydC5qcyIsICJ2aXRlL3BsdWdpbnMvc3ZnLWljb24uanMiLCAidml0ZS9wbHVnaW5zL2NvbXByZXNzaW9uLmpzIiwgInZpdGUvcGx1Z2lucy9zZXR1cC1leHRlbmQuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9JOi9wcm9qZWN0cy9qYXZhLXZ1ZS9SdW9ZaS1WdWUtbWFzdGVyL3J1b3lpLXVpL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgZGVmaW5lQ29uZmlnLCBsb2FkRW52IH0gZnJvbSAndml0ZSdcclxuaW1wb3J0IHBhdGggZnJvbSAncGF0aCdcclxuaW1wb3J0IGNyZWF0ZVZpdGVQbHVnaW5zIGZyb20gJy4vdml0ZS9wbHVnaW5zJ1xyXG5cclxuZXhwb3J0IGNvbnN0IGJhc2VVcmwgPSAnaHR0cDovL2xvY2FsaG9zdDo4MDgwJyAvLyBcdTU0MEVcdTdBRUZcdTYzQTVcdTUzRTNcclxuXHJcbi8vIGh0dHBzOi8vdml0ZWpzLmRldi9jb25maWcvXHJcbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZygoeyBtb2RlLCBjb21tYW5kIH0pID0+IHtcclxuICBjb25zdCBlbnYgPSBsb2FkRW52KG1vZGUsIHByb2Nlc3MuY3dkKCkpXHJcbiAgY29uc3QgeyBWSVRFX0FQUF9FTlYgfSA9IGVudlxyXG4gIHJldHVybiB7XHJcbiAgICAvLyBcdTkwRThcdTdGNzJcdTc1MUZcdTRFQTdcdTczQUZcdTU4ODNcdTU0OENcdTVGMDBcdTUzRDFcdTczQUZcdTU4ODNcdTRFMEJcdTc2ODRVUkxcdTMwMDJcclxuICAgIC8vIFx1OUVEOFx1OEJBNFx1NjBDNVx1NTFCNVx1NEUwQlx1RkYwQ3ZpdGUgXHU0RjFBXHU1MDQ3XHU4QkJFXHU0RjYwXHU3Njg0XHU1RTk0XHU3NTI4XHU2NjJGXHU4OEFCXHU5MEU4XHU3RjcyXHU1NzI4XHU0RTAwXHU0RTJBXHU1N0RGXHU1NDBEXHU3Njg0XHU2ODM5XHU4REVGXHU1Rjg0XHU0RTBBXHJcbiAgICAvLyBcdTRGOEJcdTU5ODIgaHR0cHM6Ly93d3cucnVveWkudmlwL1x1MzAwMlx1NTk4Mlx1Njc5Q1x1NUU5NFx1NzUyOFx1ODhBQlx1OTBFOFx1N0Y3Mlx1NTcyOFx1NEUwMFx1NEUyQVx1NUI1MFx1OERFRlx1NUY4NFx1NEUwQVx1RkYwQ1x1NEY2MFx1NUMzMVx1OTcwMFx1ODk4MVx1NzUyOFx1OEZEOVx1NEUyQVx1OTAwOVx1OTg3OVx1NjMwN1x1NUI5QVx1OEZEOVx1NEUyQVx1NUI1MFx1OERFRlx1NUY4NFx1MzAwMlx1NEY4Qlx1NTk4Mlx1RkYwQ1x1NTk4Mlx1Njc5Q1x1NEY2MFx1NzY4NFx1NUU5NFx1NzUyOFx1ODhBQlx1OTBFOFx1N0Y3Mlx1NTcyOCBodHRwczovL3d3dy5ydW95aS52aXAvYWRtaW4vXHVGRjBDXHU1MjE5XHU4QkJFXHU3RjZFIGJhc2VVcmwgXHU0RTNBIC9hZG1pbi9cdTMwMDJcclxuICAgIGJhc2U6IFZJVEVfQVBQX0VOViA9PT0gJ3Byb2R1Y3Rpb24nID8gJy8nIDogJy8nLFxyXG4gICAgcGx1Z2luczogY3JlYXRlVml0ZVBsdWdpbnMoZW52LCBjb21tYW5kID09PSAnYnVpbGQnKSxcclxuICAgIHJlc29sdmU6IHtcclxuICAgICAgLy8gaHR0cHM6Ly9jbi52aXRlanMuZGV2L2NvbmZpZy8jcmVzb2x2ZS1hbGlhc1xyXG4gICAgICBhbGlhczoge1xyXG4gICAgICAgIC8vIFx1OEJCRVx1N0Y2RVx1OERFRlx1NUY4NFxyXG4gICAgICAgICd+JzogcGF0aC5yZXNvbHZlKF9fZGlybmFtZSwgJy4vJyksXHJcbiAgICAgICAgLy8gXHU4QkJFXHU3RjZFXHU1MjJCXHU1NDBEXHJcbiAgICAgICAgJ0AnOiBwYXRoLnJlc29sdmUoX19kaXJuYW1lLCAnLi9zcmMnKVxyXG4gICAgICB9LFxyXG4gICAgICAvLyBodHRwczovL2NuLnZpdGVqcy5kZXYvY29uZmlnLyNyZXNvbHZlLWV4dGVuc2lvbnNcclxuICAgICAgZXh0ZW5zaW9uczogWycubWpzJywgJy5qcycsICcudHMnLCAnLmpzeCcsICcudHN4JywgJy5qc29uJywgJy52dWUnXVxyXG4gICAgfSxcclxuICAgIC8vIFx1NjI1M1x1NTMwNVx1OTE0RFx1N0Y2RVxyXG4gICAgYnVpbGQ6IHtcclxuICAgICAgLy8gaHR0cHM6Ly92aXRlLmRldi9jb25maWcvYnVpbGQtb3B0aW9ucy5odG1sXHJcbiAgICAgIHNvdXJjZW1hcDogY29tbWFuZCA9PT0gJ2J1aWxkJyA/IGZhbHNlIDogJ2lubGluZScsXHJcbiAgICAgIG91dERpcjogJ2Rpc3QnLFxyXG4gICAgICBhc3NldHNEaXI6ICdhc3NldHMnLFxyXG4gICAgICBjaHVua1NpemVXYXJuaW5nTGltaXQ6IDIwMDAsXHJcbiAgICAgIHJvbGx1cE9wdGlvbnM6IHtcclxuICAgICAgICBvdXRwdXQ6IHtcclxuICAgICAgICAgIGNodW5rRmlsZU5hbWVzOiAnc3RhdGljL2pzL1tuYW1lXS1baGFzaF0uanMnLFxyXG4gICAgICAgICAgZW50cnlGaWxlTmFtZXM6ICdzdGF0aWMvanMvW25hbWVdLVtoYXNoXS5qcycsXHJcbiAgICAgICAgICBhc3NldEZpbGVOYW1lczogJ3N0YXRpYy9bZXh0XS9bbmFtZV0tW2hhc2hdLltleHRdJ1xyXG4gICAgICAgIH1cclxuICAgICAgfVxyXG4gICAgfSxcclxuICAgIC8vIHZpdGUgXHU3NkY4XHU1MTczXHU5MTREXHU3RjZFXHJcbiAgICBzZXJ2ZXI6IHtcclxuICAgICAgcG9ydDogODAsXHJcbiAgICAgIGhvc3Q6IHRydWUsXHJcbiAgICAgIG9wZW46IHRydWUsXHJcbiAgICAgIHByb3h5OiB7XHJcbiAgICAgICAgLy8gaHR0cHM6Ly9jbi52aXRlanMuZGV2L2NvbmZpZy8jc2VydmVyLXByb3h5XHJcbiAgICAgICAgJy9kZXYtYXBpJzoge1xyXG4gICAgICAgICAgdGFyZ2V0OiBiYXNlVXJsLFxyXG4gICAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxyXG4gICAgICAgICAgcmV3cml0ZTogKHApID0+IHAucmVwbGFjZSgvXlxcL2Rldi1hcGkvLCAnJylcclxuICAgICAgICB9LFxyXG4gICAgICAgICAvLyBzcHJpbmdkb2MgcHJveHlcclxuICAgICAgICAgJ14vdjMvYXBpLWRvY3MvKC4qKSc6IHtcclxuICAgICAgICAgIHRhcmdldDogYmFzZVVybCxcclxuICAgICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcclxuICAgICAgICB9XHJcbiAgICAgIH1cclxuICAgIH0sXHJcbiAgICBjc3M6IHtcclxuICAgICAgcG9zdGNzczoge1xyXG4gICAgICAgIHBsdWdpbnM6IFtcclxuICAgICAgICAgIHtcclxuICAgICAgICAgICAgcG9zdGNzc1BsdWdpbjogJ2ludGVybmFsOmNoYXJzZXQtcmVtb3ZhbCcsXHJcbiAgICAgICAgICAgIEF0UnVsZToge1xyXG4gICAgICAgICAgICAgIGNoYXJzZXQ6IChhdFJ1bGUpID0+IHtcclxuICAgICAgICAgICAgICAgIGlmIChhdFJ1bGUubmFtZSA9PT0gJ2NoYXJzZXQnKSB7XHJcbiAgICAgICAgICAgICAgICAgIGF0UnVsZS5yZW1vdmUoKVxyXG4gICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgfVxyXG4gICAgICAgICAgfVxyXG4gICAgICAgIF1cclxuICAgICAgfVxyXG4gICAgfVxyXG4gIH1cclxufSlcclxuIiwgImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGVcXFxccGx1Z2luc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiSTpcXFxccHJvamVjdHNcXFxcamF2YS12dWVcXFxcUnVvWWktVnVlLW1hc3RlclxcXFxydW95aS11aVxcXFx2aXRlXFxcXHBsdWdpbnNcXFxcaW5kZXguanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0k6L3Byb2plY3RzL2phdmEtdnVlL1J1b1lpLVZ1ZS1tYXN0ZXIvcnVveWktdWkvdml0ZS9wbHVnaW5zL2luZGV4LmpzXCI7aW1wb3J0IHZ1ZSBmcm9tICdAdml0ZWpzL3BsdWdpbi12dWUnXHJcblxyXG5pbXBvcnQgY3JlYXRlQXV0b0ltcG9ydCBmcm9tICcuL2F1dG8taW1wb3J0J1xyXG5pbXBvcnQgY3JlYXRlU3ZnSWNvbiBmcm9tICcuL3N2Zy1pY29uJ1xyXG5pbXBvcnQgY3JlYXRlQ29tcHJlc3Npb24gZnJvbSAnLi9jb21wcmVzc2lvbidcclxuaW1wb3J0IGNyZWF0ZVNldHVwRXh0ZW5kIGZyb20gJy4vc2V0dXAtZXh0ZW5kJ1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gY3JlYXRlVml0ZVBsdWdpbnModml0ZUVudiwgaXNCdWlsZCA9IGZhbHNlKSB7XHJcbiAgICBjb25zdCB2aXRlUGx1Z2lucyA9IFt2dWUoKV1cclxuICAgIHZpdGVQbHVnaW5zLnB1c2goY3JlYXRlQXV0b0ltcG9ydCgpKVxyXG5cdHZpdGVQbHVnaW5zLnB1c2goY3JlYXRlU2V0dXBFeHRlbmQoKSlcclxuICAgIHZpdGVQbHVnaW5zLnB1c2goY3JlYXRlU3ZnSWNvbihpc0J1aWxkKSlcclxuXHRpc0J1aWxkICYmIHZpdGVQbHVnaW5zLnB1c2goLi4uY3JlYXRlQ29tcHJlc3Npb24odml0ZUVudikpXHJcbiAgICByZXR1cm4gdml0ZVBsdWdpbnNcclxufVxyXG4iLCAiY29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2Rpcm5hbWUgPSBcIkk6XFxcXHByb2plY3RzXFxcXGphdmEtdnVlXFxcXFJ1b1lpLVZ1ZS1tYXN0ZXJcXFxccnVveWktdWlcXFxcdml0ZVxcXFxwbHVnaW5zXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGVcXFxccGx1Z2luc1xcXFxhdXRvLWltcG9ydC5qc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vSTovcHJvamVjdHMvamF2YS12dWUvUnVvWWktVnVlLW1hc3Rlci9ydW95aS11aS92aXRlL3BsdWdpbnMvYXV0by1pbXBvcnQuanNcIjtpbXBvcnQgYXV0b0ltcG9ydCBmcm9tICd1bnBsdWdpbi1hdXRvLWltcG9ydC92aXRlJ1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gY3JlYXRlQXV0b0ltcG9ydCgpIHtcclxuICAgIHJldHVybiBhdXRvSW1wb3J0KHtcclxuICAgICAgICBpbXBvcnRzOiBbXHJcbiAgICAgICAgICAgICd2dWUnLFxyXG4gICAgICAgICAgICAndnVlLXJvdXRlcicsXHJcbiAgICAgICAgICAgICdwaW5pYSdcclxuICAgICAgICBdLFxyXG4gICAgICAgIGR0czogZmFsc2VcclxuICAgIH0pXHJcbn1cclxuIiwgImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGVcXFxccGx1Z2luc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiSTpcXFxccHJvamVjdHNcXFxcamF2YS12dWVcXFxcUnVvWWktVnVlLW1hc3RlclxcXFxydW95aS11aVxcXFx2aXRlXFxcXHBsdWdpbnNcXFxcc3ZnLWljb24uanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0k6L3Byb2plY3RzL2phdmEtdnVlL1J1b1lpLVZ1ZS1tYXN0ZXIvcnVveWktdWkvdml0ZS9wbHVnaW5zL3N2Zy1pY29uLmpzXCI7aW1wb3J0IHsgY3JlYXRlU3ZnSWNvbnNQbHVnaW4gfSBmcm9tICd2aXRlLXBsdWdpbi1zdmctaWNvbnMnXHJcbmltcG9ydCBwYXRoIGZyb20gJ3BhdGgnXHJcblxyXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBjcmVhdGVTdmdJY29uKGlzQnVpbGQpIHtcclxuICAgIHJldHVybiBjcmVhdGVTdmdJY29uc1BsdWdpbih7XHJcblx0XHRpY29uRGlyczogW3BhdGgucmVzb2x2ZShwcm9jZXNzLmN3ZCgpLCAnc3JjL2Fzc2V0cy9pY29ucy9zdmcnKV0sXHJcbiAgICAgICAgc3ltYm9sSWQ6ICdpY29uLVtkaXJdLVtuYW1lXScsXHJcbiAgICAgICAgc3Znb09wdGlvbnM6IGlzQnVpbGRcclxuICAgIH0pXHJcbn1cclxuIiwgImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGVcXFxccGx1Z2luc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiSTpcXFxccHJvamVjdHNcXFxcamF2YS12dWVcXFxcUnVvWWktVnVlLW1hc3RlclxcXFxydW95aS11aVxcXFx2aXRlXFxcXHBsdWdpbnNcXFxcY29tcHJlc3Npb24uanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0k6L3Byb2plY3RzL2phdmEtdnVlL1J1b1lpLVZ1ZS1tYXN0ZXIvcnVveWktdWkvdml0ZS9wbHVnaW5zL2NvbXByZXNzaW9uLmpzXCI7aW1wb3J0IGNvbXByZXNzaW9uIGZyb20gJ3ZpdGUtcGx1Z2luLWNvbXByZXNzaW9uJ1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gY3JlYXRlQ29tcHJlc3Npb24oZW52KSB7XHJcbiAgICBjb25zdCB7IFZJVEVfQlVJTERfQ09NUFJFU1MgfSA9IGVudlxyXG4gICAgY29uc3QgcGx1Z2luID0gW11cclxuICAgIGlmIChWSVRFX0JVSUxEX0NPTVBSRVNTKSB7XHJcbiAgICAgICAgY29uc3QgY29tcHJlc3NMaXN0ID0gVklURV9CVUlMRF9DT01QUkVTUy5zcGxpdCgnLCcpXHJcbiAgICAgICAgaWYgKGNvbXByZXNzTGlzdC5pbmNsdWRlcygnZ3ppcCcpKSB7XHJcbiAgICAgICAgICAgIC8vIGh0dHA6Ly9kb2MucnVveWkudmlwL3J1b3lpLXZ1ZS9vdGhlci9mYXEuaHRtbCNcdTRGN0ZcdTc1MjhnemlwXHU4OUUzXHU1MzhCXHU3RjI5XHU5NzU5XHU2MDAxXHU2NTg3XHU0RUY2XHJcbiAgICAgICAgICAgIHBsdWdpbi5wdXNoKFxyXG4gICAgICAgICAgICAgICAgY29tcHJlc3Npb24oe1xyXG4gICAgICAgICAgICAgICAgICAgIGV4dDogJy5neicsXHJcbiAgICAgICAgICAgICAgICAgICAgZGVsZXRlT3JpZ2luRmlsZTogZmFsc2VcclxuICAgICAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICAgIClcclxuICAgICAgICB9XHJcbiAgICAgICAgaWYgKGNvbXByZXNzTGlzdC5pbmNsdWRlcygnYnJvdGxpJykpIHtcclxuICAgICAgICAgICAgcGx1Z2luLnB1c2goXHJcbiAgICAgICAgICAgICAgICBjb21wcmVzc2lvbih7XHJcbiAgICAgICAgICAgICAgICAgICAgZXh0OiAnLmJyJyxcclxuICAgICAgICAgICAgICAgICAgICBhbGdvcml0aG06ICdicm90bGlDb21wcmVzcycsXHJcbiAgICAgICAgICAgICAgICAgICAgZGVsZXRlT3JpZ2luRmlsZTogZmFsc2VcclxuICAgICAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICAgIClcclxuICAgICAgICB9XHJcbiAgICB9XHJcbiAgICByZXR1cm4gcGx1Z2luXHJcbn1cclxuIiwgImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJJOlxcXFxwcm9qZWN0c1xcXFxqYXZhLXZ1ZVxcXFxSdW9ZaS1WdWUtbWFzdGVyXFxcXHJ1b3lpLXVpXFxcXHZpdGVcXFxccGx1Z2luc1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiSTpcXFxccHJvamVjdHNcXFxcamF2YS12dWVcXFxcUnVvWWktVnVlLW1hc3RlclxcXFxydW95aS11aVxcXFx2aXRlXFxcXHBsdWdpbnNcXFxcc2V0dXAtZXh0ZW5kLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9JOi9wcm9qZWN0cy9qYXZhLXZ1ZS9SdW9ZaS1WdWUtbWFzdGVyL3J1b3lpLXVpL3ZpdGUvcGx1Z2lucy9zZXR1cC1leHRlbmQuanNcIjtpbXBvcnQgc2V0dXBFeHRlbmQgZnJvbSAndW5wbHVnaW4tdnVlLXNldHVwLWV4dGVuZC1wbHVzL3ZpdGUnXHJcblxyXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBjcmVhdGVTZXR1cEV4dGVuZCgpIHtcclxuICAgIHJldHVybiBzZXR1cEV4dGVuZCh7fSlcclxufVxyXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQXNVLFNBQVMsY0FBYyxlQUFlO0FBQzVXLE9BQU9BLFdBQVU7OztBQ0RvVixPQUFPLFNBQVM7OztBQ0FKLE9BQU8sZ0JBQWdCO0FBRXpYLFNBQVIsbUJBQW9DO0FBQ3ZDLFNBQU8sV0FBVztBQUFBLElBQ2Qs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
