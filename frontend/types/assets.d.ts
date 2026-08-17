/**
 * Side-effect imports of stylesheets.
 *
 * TypeScript 7's native checker requires a declaration for a side-effect import
 * (`import "./globals.css"`), where 5.x silently allowed it. Next's own
 * next-env.d.ts does not declare these, so state them here rather than loosening
 * the compiler.
 */
declare module "*.css";
declare module "*.scss";
