import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";

import DefaultNav from "../components/DefaultNav";

import "../styles/normalize.css";
import "../styles/prismjs.css";
import "../styles/base.css";

const Wrapper = styled.div`
  max-width: 960px;
  margin: auto;
`;

class Template extends React.Component {
  render() {
    const { location, children } = this.props;
    let header;
    if (location.pathname === "/") {
      header = <div />;
    } else {
      header = <DefaultNav />;
    }
    return (
      <div>
        <Wrapper>
          {header}
          {children()}
        </Wrapper>
      </div>
    );
  }
}

export default Template;
