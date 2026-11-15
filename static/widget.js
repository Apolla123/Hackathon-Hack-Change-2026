/* Встраивание консультанта в сайт Energytool: одна строка перед </body>
   <script src="https://ВАШ-ДОМЕН/static/widget.js" async></script>
   Виджет добавляет плавающую кнопку и открывает чат в iframe (режим ?embed=1, без боковой панели). */
(function () {
  var origin = new URL(document.currentScript.src).origin;
  var btn = document.createElement("button");
  btn.textContent = "Консультант Energytool";
  btn.setAttribute("style", "position:fixed;right:20px;bottom:20px;z-index:2147483000;background:#e30611;color:#fff;border:0;border-radius:999px;padding:14px 20px;font:600 15px system-ui;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,.25)");
  var frame = document.createElement("iframe");
  frame.src = origin + "/?embed=1";
  frame.title = "ИИ-консультант МТС Energytool";
  frame.setAttribute("style", "display:none;position:fixed;right:20px;bottom:80px;width:min(400px,calc(100vw - 40px));height:min(620px,calc(100vh - 110px));z-index:2147483000;border:1px solid #ddd;border-radius:14px;background:#fff;box-shadow:0 8px 32px rgba(0,0,0,.3)");
  btn.onclick = function () { frame.style.display = frame.style.display === "none" ? "block" : "none"; };
  document.body.appendChild(frame); document.body.appendChild(btn);
})();
